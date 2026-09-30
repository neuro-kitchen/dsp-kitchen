//! Tucker-Davis Technologies (TDT) blocks written by Synapse or OpenEx.
//!
//! A block folder holds a TSQ index, a TEV data file (or per-channel SEV files), and text
//! sidecars. Every store becomes part of one [`Session`]:
//! streams → [`Recording`](crate::model::Recording)s, epocs and scalars →
//! [`EventSeries`](crate::model::EventSeries), snips →
//! [`SnippetSeries`](crate::model::SnippetSeries), CSV exports → tables.

pub mod block;
pub mod codes;
pub mod epocs;
pub mod impedance;
pub mod notes;
pub mod sev;
pub mod snips;
pub mod sort;
pub mod streams;
pub mod tbk;
pub mod tsq;
pub mod version;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use block::BlockFiles;
use codes::StoreKind;
use notes::synapse::{parse_notes, parse_stores_listing};
use tsq::TsqIndex;
use version::TdtVersion;

use crate::common::mapped::MappedFile;
use crate::common::text::read_text;
use crate::common::time::{format_iso, parse_iso};
use crate::error::{Error, Result};
use crate::model::{Device, Provenance, Session};
use crate::options::OpenOptions;
use crate::registry::{Detection, InputFormat};

pub struct Tdt;

impl InputFormat for Tdt {
    fn name(&self) -> &'static str {
        "tdt"
    }

    fn description(&self) -> &'static str {
        "Tucker-Davis Technologies block (TSQ + TEV, Synapse or OpenEx)"
    }

    fn versions(&self) -> &'static [&'static str] {
        &["Synapse (Notes.txt, StoresListing.txt, .tin)", "OpenEx (.tnt)", "TEV streams, snips, epocs, scalars"]
    }

    fn detect(&self, path: &Path) -> Option<Detection> {
        if BlockFiles::find(path).is_some() {
            return Some(Detection { format: "tdt", version: None, confidence: 0.95 });
        }
        let blocks = block::tank_blocks(path);
        (!blocks.is_empty()).then(|| Detection { format: "tdt", version: Some(format!("tank, {} blocks", blocks.len())), confidence: 0.9 })
    }

    fn open(&self, path: &Path, options: &OpenOptions) -> Result<Session> {
        if BlockFiles::find(path).is_some() {
            return open_block(path, options);
        }
        // A tank: one of its blocks
        let blocks = block::tank_blocks(path);
        let names: Vec<String> = blocks.iter().filter_map(|b| Some(b.file_name()?.to_string_lossy().into_owned())).collect();
        let chosen = match (&options.block, blocks.len()) {
            (Some(name), _) => blocks.iter().find(|b| b.file_name().is_some_and(|n| n.to_string_lossy() == *name)),
            (None, 1) => blocks.first(),
            _ => None,
        };
        match chosen {
            Some(b) => open_block(b, options),
            None => Err(Error::Unsupported(format!(
                "{} is a TDT tank with {} blocks; choose one with --block <name>: {}",
                path.display(),
                blocks.len(),
                names.join(", ")
            ))),
        }
    }
}

/// Reads a block folder (or any file inside it) into a [`Session`].
pub fn open_block(path: &Path, options: &OpenOptions) -> Result<Session> {
    let files = BlockFiles::find(path).ok_or_else(|| Error::format("tdt", format!("no .tsq file in {}", path.display())))?;
    let tsq_map = MappedFile::open(&files.tsq)?;
    let index = TsqIndex::parse(tsq_map.bytes());
    drop(tsq_map);

    let mut warnings = index.warnings.clone();
    let listing = files.stores_listing.as_deref().and_then(read_text).map(|t| parse_stores_listing(&t)).unwrap_or_default();
    let synapse_notes = files.notes.as_deref().and_then(read_text).map(|t| parse_notes(&t));
    let tnt = files.tnt.as_deref().and_then(read_text).map(|t| notes::openex::parse_tnt(&t));
    let tin = files.tin.as_deref().and_then(notes::tin::read_tin);
    let version = TdtVersion::detect(
        files.notes.is_some() || files.stores_listing.is_some() || files.tin.is_some(),
        tin.as_ref().and_then(|t| t.get("Versions.Synapse").map(str::to_string)),
        tnt.as_ref().and_then(|t| t.version.clone()),
    );
    // Tbk store settings: expected sample rates (SEV headers can be wrong)
    let tbk = files.tbk.as_deref().and_then(|p| std::fs::read(p).ok()).map(|b| tbk::parse_tbk(&b)).unwrap_or_default();
    let expected_rate = |store: &str| {
        tbk.iter().find(|s| s.get("StoreName").map(String::as_str) == Some(store)).and_then(|s| s.get("SampleFreq")?.parse::<f64>().ok()).filter(|f| *f > 0.0)
    };
    let rawpacked = |store: &str| {
        tbk.iter().any(|s| s.get("StoreName").map(String::as_str) == Some(store) && s.get("DataFormat").map(String::as_str) == Some("8"))
    };
    let sev_stores = sev::group(&files.sev, &mut warnings);
    for log_path in std::fs::read_dir(&files.dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with("_log.txt")) {
        let name = log_path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let log = read_text(&log_path).map(|t| sev::log::parse(&name, &t)).unwrap_or_default();
        if log.start_sample.is_some_and(|s| s > 2) && log.hour == 0 {
            warnings.push(format!("{}: SEV recording starts at sample {}", log.store, log.start_sample.unwrap()));
        }
        for (a, b) in &log.gaps {
            warnings.push(format!("{}: gap in SEV data between samples {a} and {b} (not filled)", log.store));
        }
    }

    let mut session = Session::default();
    let block_end = index.stop.map_or(0.0, |s| s - index.start);

    // Streams and snips read from the TEV
    let needs_tev = index.stores.values().any(|s| {
        options.wants(&s.name) && (s.kind == StoreKind::Snip || (s.kind == StoreKind::Stream && !sev_stores.contains_key(&s.name)))
    });
    // Offline sorts under sort/<id>/<store>.SortResult
    let sorts = sort::available(&files.dir);
    if let Some(id) = options.sort.as_deref().filter(|id| !sorts.contains_key(*id)) {
        warnings.push(format!("sort {id:?} not found (available: {:?}); online sort codes kept", sorts.keys().collect::<Vec<_>>()));
    }
    let chosen_sort = |store: &str| -> Option<(String, Vec<u8>)> {
        let id = options.sort.as_ref()?;
        let file = sorts.get(id)?.get(store)?;
        sort::load(file).ok().map(|codes| (id.clone(), codes))
    };
    let tev = match (&files.tev, needs_tev) {
        (Some(p), true) => Some(Arc::new(MappedFile::open(p)?)),
        (None, true) => {
            warnings.push("stream / snip stores are indexed but the .tev file is missing".into());
            None
        }
        _ => None,
    };
    for store in index.stores.values().filter(|s| options.wants(&s.name)) {
        match (store.kind, &tev) {
            // SEV files win over TEV packets for the same store
            (StoreKind::Stream, _) if sev_stores.contains_key(&store.name) => {}
            (StoreKind::Stream, Some(tev)) => {
                match streams::TdtStream::new(store, tev.clone(), index.start, listing.stores.get(&store.name), &mut warnings) {
                    Ok(s) => session.recordings.push(Arc::new(s)),
                    Err(e) => warnings.push(format!("{}: skipped ({e})", store.name)),
                }
            }
            (StoreKind::Snip, Some(tev)) => match snips::build(store, tev, index.start, chosen_sort(&store.name).as_ref().map(|(id, c)| (id.as_str(), c.as_slice())), &mut warnings) {
                Ok(s) => session.snippets.push(s),
                Err(e) => warnings.push(format!("{}: skipped ({e})", store.name)),
            },
            _ => {}
        }
    }
    // SEV streams (RS4 stores never appear in the TSQ)
    for (name, channels) in sev_stores.iter().filter(|(n, _)| options.wants(n)) {
        let start = index.stores.get(name).and_then(|s| s.first_timestamp.values().copied().reduce(f64::min)).map_or(0.0, |t| tsq::session_time(t, index.start));
        let description = listing.stores.get(name).map_or_else(|| "TDT stream store (SEV files)".to_string(), |d| format!("{} ({})", d.object, d.object_type));
        match sev::SevStream::build(name, channels, start.max(0.0), expected_rate(name), rawpacked(name), description, &mut warnings) {
            Ok(v) => session.recordings.extend(v.into_iter().map(|r| Arc::new(r) as Arc<dyn crate::model::Recording>)),
            Err(e) => warnings.push(format!("{name}: SEV skipped ({e})")),
        }
    }
    for s in index.stores.values().filter(|s| s.kind == StoreKind::Stream && s.evtype & codes::EVTYPE_UCF != 0) {
        if !sev_stores.contains_key(&s.name) {
            warnings.push(format!("{}: expected SEV files (stored unscaled) but none were found", s.name));
        }
    }
    session.recordings.sort_by(|a, b| a.info().name.cmp(&b.info().name));
    let wanted: BTreeMap<String, tsq::StoreIndex> =
        index.stores.iter().filter(|(n, _)| options.wants(n) || options.wants(&n.replace('\\', "/"))).map(|(n, s)| (n.clone(), s.clone())).collect();
    session.events = epocs::build(&wanted, index.start, block_end, &listing.stores, &mut warnings);
    if let Some(n) = synapse_notes.as_ref().filter(|_| options.wants("Note")) {
        epocs::attach_notes(&mut session.events, n, &mut warnings);
    }
    session.tables = files.csv.iter().filter_map(|p| impedance::read_csv_table(p)).collect();

    // Metadata: .tin summary first (exact ISO start), then Notes.txt, then the TSQ clock
    let m = &mut session.metadata;
    let field = |tin_key: &str, note_key: &str| {
        tin.as_ref()
            .and_then(|t| t.get(tin_key).map(str::to_string))
            .or_else(|| synapse_notes.as_ref().and_then(|n| n.fields.get(note_key).cloned()))
    };
    m.experiment = field("Experiment.Name", "Experiment");
    m.subject.id = field("Subject.Name", "Subject");
    m.experimenters = field("User.Name", "User").into_iter().collect();
    let local_start = tin.as_ref().and_then(|t| t.get("Recording.StartTime")).and_then(parse_iso);
    m.start_time = Some(local_start.map_or_else(|| format!("{}Z", format_iso(index.start)), format_iso));
    if let Some(stop) = index.stop {
        let duration = stop - index.start;
        m.stop_time = Some(local_start.map_or_else(|| format!("{}Z", format_iso(stop)), |s| format_iso(s + duration)));
    }
    m.devices = listing
        .hardware
        .iter()
        .map(|(name, ty)| Device {
            name: name.clone(),
            description: ty.clone(),
            manufacturer: Some("Tucker-Davis Technologies".into()),
            model: hardware_model(name),
        })
        .collect();
    m.notes.extend(synapse_notes.iter().flat_map(|n| n.notes.clone()));
    m.notes.extend(tnt.iter().flat_map(|t| t.notes.clone()));
    m.extra.insert("tdt_block".into(), files.name.clone());
    if !sorts.is_empty() {
        m.extra.insert("tdt_sorts".into(), sorts.keys().cloned().collect::<Vec<_>>().join(", "));
    }
    if let Some(id) = options.sort.as_ref().filter(|id| sorts.contains_key(*id)) {
        m.extra.insert("tdt_sort_applied".into(), id.clone());
    }
    m.extra.insert("tdt_software".into(), version.to_string());
    m.extra.insert("tdt_start_unix".into(), format!("{:.6}", index.start));
    if let Some(stop) = index.stop {
        m.extra.insert("tdt_stop_unix".into(), format!("{stop:.6}"));
    }
    if let Some(t) = &tin {
        for (k, v) in t.fields.iter().filter(|(k, _)| k.starts_with("Recording.") || k.as_str() == "Experiment.GUID") {
            m.extra.insert(format!("synapse_{}", k.to_lowercase().replace('.', "_")), v.clone());
        }
    }

    let mut prov = Provenance::new("tdt");
    prov.version = Some(version.to_string());
    for f in files.all() {
        prov.add_file(f);
    }
    prov.warnings = warnings;
    session.provenance = prov;
    Ok(session)
}

/// Model of a Synapse hardware object from its id: Synapse names instances `MODEL(n)`
/// (`RZ2(1)`, `IZV10(1)`, `PZ5(2)`).
fn hardware_model(object: &str) -> Option<String> {
    let (model, rest) = object.split_once('(')?;
    let index = rest.strip_suffix(')')?;
    (!model.is_empty() && !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit())).then(|| model.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsq::tests::record;

    /// A tiny Synapse-like block: one 2-channel float stream (4 samples per packet), one epoc.
    fn write_block(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let mut tev = Vec::new();
        let mut tsq = record(0, 0, &[0; 4], 0, 0, 0.0, 0, 0, 0.0);
        tsq.extend(record(10, codes::EVTYPE_MARK, &1u32.to_le_bytes(), 0, 0, 1000.0, 0, 0, 0.0));
        // Packets in time order, channels interleaved: values = ch * 100 + sample
        for k in 0..3u64 {
            for ch in 1..=2u16 {
                let off = tev.len() as u64;
                for s in 0..4 {
                    tev.extend((ch as f32 * 100.0 + (k * 4 + s) as f32).to_le_bytes());
                }
                tsq.extend(record(14, codes::EVTYPE_STREAM, b"Wav1", ch, 0, 1000.0 + k as f64 * 0.04, off, 0, 100.0));
            }
        }
        tsq.extend(record(10, codes::EVTYPE_STRON, b"Tick", 0, 0, 1000.5, 1f64.to_bits(), 4, 0.0));
        tsq.extend(record(10, codes::EVTYPE_MARK, &2u32.to_le_bytes(), 0, 0, 1001.0, 0, 0, 0.0));
        std::fs::write(dir.join("t_b.tsq"), tsq).unwrap();
        std::fs::write(dir.join("t_b.tev"), tev).unwrap();
        std::fs::write(dir.join("t_b.tnt"), "NOTEFILE_VERSION[1.0]\r\n").unwrap();
    }

    /// SEV cases: v3 float32 with an hour split, rawpacked, headerless v0, a wrong header rate
    /// corrected by the Tbk, and SEV winning over TEV packets of the same store.
    /// Written to `target/tdt-sev-fixture` so TDT's Python reader can cross-check it.
    #[test]
    fn test_sev_block() {
        use sev::header::tests::header;
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tdt-sev-fixture");
        let _ = std::fs::remove_dir_all(&dir);
        write_block(&dir);
        let sev = |name: &str, head: Vec<u8>, payload: Vec<u8>| {
            let mut b = head;
            b.extend(payload);
            std::fs::write(dir.join(format!("t_b_{name}.sev")), b).unwrap();
        };
        let f32s = |v: std::ops::Range<i32>, add: f32| v.flat_map(|x| (x as f32 + add).to_le_bytes()).collect::<Vec<u8>>();
        // Wav1 (also in the TSQ/TEV above): SEV v3, rate code 2 / decimate 1 = 24414.0625 Hz,
        // but the Tbk says 100 Hz; ch1 split into two hour files
        sev("Wav1_Ch1", header(3, b"Wav1", 1, 2, 4, 0, 1, 2), f32s(0..6, 1000.0));
        sev("Wav1_Ch1-1h", header(3, b"Wav1", 1, 2, 4, 0, 1, 2), f32s(6..12, 1000.0));
        sev("Wav1_Ch2", header(3, b"Wav1", 2, 2, 4, 0, 1, 2), f32s(0..12, 2000.0));
        std::fs::write(
            dir.join("t_b.Tbk"),
            "x[USERNOTEDELIMITER]y[USERNOTEDELIMITER]NAME=StoreName;TYPE=T;VALUE=Wav1;\nNAME=SampleFreq;TYPE=L;VALUE=100;\n\
[STOREHDRITEM]NAME=StoreName;TYPE=T;VALUE=RSn1;\nNAME=DataFormat;TYPE=L;VALUE=8;\n[USERNOTEDELIMITER]",
        )
        .unwrap();
        // RSn1: rawpacked (declared by the Tbk) words: single-unit = high 16 bits, LFP = low 16 bits
        let words: Vec<u8> = [(-3i32 << 16) | 7, (5 << 16) | 0xFFFF].iter().flat_map(|w| w.to_le_bytes()).collect();
        sev("RSn1_Ch1", header(3, b"RSn1", 1, 1, 4, 8, 1, 2), words);
        // Old1: headerless v0 file (float32 assumed), store and channel from the name
        sev("Old1_Ch3", vec![0u8; 40], f32s(0..4, 0.5));

        let s = open_block(&dir, &OpenOptions::default()).unwrap();
        let names: Vec<String> = s.recordings.iter().map(|r| r.info().name.clone()).collect();
        assert_eq!(names, vec!["Old1", "RSn1_LFP", "RSn1_SU", "Wav1"]);

        let w = s.recording("Wav1").unwrap();
        assert_eq!((w.info().samples, w.info().sample_rate), (12, 100.0), "hour files joined, Tbk rate wins");
        let mut out = vec![0.0; 2 * 4];
        w.read(&[0, 1], 4..8, &mut out).unwrap();
        assert_eq!(out, vec![1004.0, 1005.0, 1006.0, 1007.0, 2004.0, 2005.0, 2006.0, 2007.0], "SEV, not TEV");

        let mut su = vec![0.0; 2];
        s.recording("RSn1_SU").unwrap().read(&[0], 0..2, &mut su).unwrap();
        let mut lfp = vec![0.0; 2];
        s.recording("RSn1_LFP").unwrap().read(&[0], 0..2, &mut lfp).unwrap();
        assert_eq!((su, lfp), (vec![-3.0, 5.0], vec![7.0, -1.0]));
        let mut raw = vec![0u8; 4];
        assert!(s.recording("RSn1_LFP").unwrap().read_stored(&[0], 0..2, &mut raw).unwrap());
        assert_eq!(raw, [7, 0, 0xFF, 0xFF]);

        let old = s.recording("Old1").unwrap();
        assert_eq!((old.info().sample_rate, old.info().channels[0].name.as_str()), (24_414.0625, "Old1 3"));
        let w = &s.provenance.warnings;
        assert!(w.iter().any(|m| m.contains("no header (v0)")), "{w:?}");
        assert!(w.iter().any(|m| m.contains("differs from the block notes")), "{w:?}");
    }

    /// A tank with two blocks; block `b1` has 3 snippets (4 float samples each) and an offline
    /// sort `Mine` giving them codes 5, 6, 7. Kept in `target/tdt-tank-fixture` so TDT's reader
    /// can cross-check the sort-code indexing.
    #[test]
    fn test_sort_result_and_tank() {
        let tank = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/tdt-tank-fixture");
        let _ = std::fs::remove_dir_all(&tank);
        let b1 = tank.join("b1");
        std::fs::create_dir_all(&b1).unwrap();
        write_block(&tank.join("b2"));

        let mut tev = Vec::new();
        let mut tsq = record(0, 0, &[0; 4], 0, 0, 0.0, 0, 0, 0.0);
        tsq.extend(record(10, codes::EVTYPE_MARK, &1u32.to_le_bytes(), 0, 0, 1000.0, 0, 0, 0.0)); // seq 0
        tsq.extend(record(10, codes::EVTYPE_STRON, b"Tick", 0, 0, 1000.1, 0, 4, 0.0)); // seq 1
        for k in 0..3u16 {
            let off = tev.len() as u64;
            for v in 0..4 {
                tev.extend((k as f32 * 10.0 + v as f32).to_le_bytes());
            }
            // seqs 2, 3, 4; online sort code 1
            tsq.extend(record(14, codes::EVTYPE_SNIP, b"eNe1", k + 1, 1, 1000.2 + k as f64 * 0.1, off, 0, 24414.0625));
        }
        tsq.extend(record(10, codes::EVTYPE_MARK, &2u32.to_le_bytes(), 0, 0, 1001.0, 0, 0, 0.0));
        std::fs::write(b1.join("t_b1.tsq"), tsq).unwrap();
        std::fs::write(b1.join("t_b1.tev"), tev).unwrap();
        std::fs::create_dir_all(b1.join("sort/Mine")).unwrap();
        let mut sort = vec![0u8; 1024];
        sort[..3].fill(1);
        sort.extend([0, 0, 5, 6, 7, 0]);
        std::fs::write(b1.join("sort/Mine/eNe1.SortResult"), sort).unwrap();

        // Tank: detected, and several blocks need a choice
        assert_eq!(Tdt.detect(&tank).unwrap().version.as_deref(), Some("tank, 2 blocks"));
        let err = Tdt.open(&tank, &OpenOptions::default()).err().unwrap().to_string();
        assert!(err.contains("--block") && err.contains("b1, b2"), "{err}");

        let online = Tdt.open(&tank, &OpenOptions { block: Some("b1".into()), ..Default::default() }).unwrap();
        let sn = &online.snippets[0];
        assert_eq!((sn.len(), sn.samples_per_snippet, sn.channels.clone()), (3, 4, vec![1, 2, 3]));
        assert_eq!(&sn.data[4..8], &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(sn.sort_codes, vec![1, 1, 1]);
        assert_eq!(online.metadata.extra["tdt_sorts"], "Mine");

        let sorted = Tdt.open(&b1, &OpenOptions { sort: Some("Mine".into()), ..Default::default() }).unwrap();
        assert_eq!(sorted.snippets[0].sort_codes, vec![5, 6, 7]);
        assert_eq!(sorted.metadata.extra["tdt_sort_applied"], "Mine");

        let missing = Tdt.open(&b1, &OpenOptions { sort: Some("Nope".into()), ..Default::default() }).unwrap();
        assert!(missing.provenance.warnings.iter().any(|w| w.contains("\"Nope\" not found")));
    }

    #[test]
    fn test_open_block_end_to_end() {
        let dir = std::env::temp_dir().join(format!("nc_tdt_{}", std::process::id()));
        write_block(&dir);
        assert!(Tdt.detect(&dir).is_some());
        let s = open_block(&dir.join("t_b.tsq"), &OpenOptions::default()).unwrap();

        assert_eq!(s.recordings.len(), 1);
        let r = &s.recordings[0];
        assert_eq!((r.info().channel_count(), r.info().samples, r.info().sample_rate), (2, 12, 100.0));
        // Crosses a packet boundary, channels reversed
        let mut out = vec![0.0; 2 * 3];
        r.read(&[1, 0], 3..6, &mut out).unwrap();
        assert_eq!(out, vec![203.0, 204.0, 205.0, 103.0, 104.0, 105.0]);

        let tick = s.event_series("Tick").unwrap();
        assert!((tick.onsets[0] - 0.5).abs() < 1.0 / 195_312.5);
        assert_eq!(s.metadata.start_time.as_deref(), Some("1970-01-01T00:16:40Z"));
        assert_eq!(s.provenance.version.as_deref(), Some("OpenEx"));
        assert!(s.provenance.warnings.is_empty(), "{:?}", s.provenance.warnings);

        let only = open_block(&dir, &OpenOptions { only: Some(vec!["Tick".into()]), ..Default::default() }).unwrap();
        assert!(only.recordings.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hardware_models_come_from_synapse_object_ids() {
        assert_eq!(hardware_model("RZ2(1)").as_deref(), Some("RZ2"));
        assert_eq!(hardware_model("IZV10(1)").as_deref(), Some("IZV10"));
        assert_eq!(hardware_model("bpPressure"), None);
        assert_eq!(hardware_model("Odd(x)"), None);
    }
}
