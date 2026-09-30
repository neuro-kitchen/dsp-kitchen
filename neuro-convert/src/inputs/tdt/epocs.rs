//! Epoc and scalar stores → [`EventSeries`].
//!
//! - Epoc onsets carry their value in the record's offset field (as `f64`).
//! - Offsets are a separate store (name ending in `\`) whose channel + sortcode fields hold the
//!   4-byte name of the onset store they close.
//! - Scalars write one record per channel with a shared timestamp; values follow the TSQ data
//!   format like TDT's own reader does.

use std::collections::BTreeMap;

use super::codes::StoreKind;
use super::notes::synapse::{clock_seconds, StoreDescription, SynapseNotes};
use super::tsq::{session_time, store_name, StoreIndex};
use crate::model::EventSeries;

/// Builds epoc series (onsets paired with their offset stores) and scalar series.
pub fn build(
    stores: &BTreeMap<String, StoreIndex>,
    block_start: f64,
    block_end: f64,
    listing: &BTreeMap<String, StoreDescription>,
    warnings: &mut Vec<String>,
) -> Vec<EventSeries> {
    // Offsets keyed by the onset store they belong to
    let mut offsets: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for s in stores.values().filter(|s| s.kind == StoreKind::EpocOffset) {
        let partner = s.records.first().map(|r| {
            let mut name = [0u8; 4];
            name[..2].copy_from_slice(&r.chan.to_le_bytes());
            name[2..].copy_from_slice(&r.sortcode.to_le_bytes());
            store_name(&name)
        });
        let partner = partner.filter(|p| stores.contains_key(p)).unwrap_or_else(|| s.name.replace('\\', "/"));
        offsets.insert(partner, s.records.iter().map(|r| session_time(r.timestamp, block_start)).collect());
    }

    let describe = |name: &str| {
        listing.get(name).map(|d| {
            let mut t = format!("{} ({})", d.object, d.object_type);
            if let Some(src) = &d.source {
                t.push_str(&format!("; {src}"));
            }
            t
        })
    };

    let mut out = Vec::new();
    for s in stores.values() {
        match s.kind {
            StoreKind::EpocOnset => {
                let onsets: Vec<f64> = s.records.iter().map(|r| session_time(r.timestamp, block_start)).collect();
                let offs = offsets.remove(&s.name).and_then(|mut off| {
                    if off.len() + 1 == onsets.len() {
                        // Still open when recording stopped
                        off.push(block_end);
                        warnings.push(format!("{}: last epoc had no offset; closed at the block end", s.name));
                    }
                    if off.len() == onsets.len() {
                        Some(off)
                    } else {
                        warnings.push(format!("{}: {} onsets but {} offsets; offsets dropped", s.name, onsets.len(), off.len()));
                        None
                    }
                });
                out.push(EventSeries {
                    name: s.name.clone(),
                    description: describe(&s.name).unwrap_or_else(|| "TDT epoc store".into()),
                    values: s.records.iter().map(|r| r.value()).collect(),
                    onsets,
                    offsets: offs,
                    channels: 1,
                    labels: Vec::new(),
                });
            }
            StoreKind::Scalar => out.push(scalars(s, block_start, describe(&s.name))),
            _ => {}
        }
    }
    for (orphan, _) in offsets {
        warnings.push(format!("offset store for {orphan} has no onset store"));
    }
    out
}

/// Attaches `Notes.txt` runtime notes as labels of the `Note` epoc (the n-th note belongs to
/// the n-th event). Without a `Note` store, one is built from the notes' wall-clock times
/// relative to `Start` (1 s resolution), as TDT's reader does.
pub fn attach_notes(events: &mut Vec<EventSeries>, notes: &SynapseNotes, warnings: &mut Vec<String>) {
    if notes.entries.is_empty() {
        return;
    }
    let labels: Vec<String> = notes.entries.iter().map(|n| n.label()).collect();
    if let Some(e) = events.iter_mut().find(|e| e.name == "Note") {
        if e.len() != labels.len() {
            warnings.push(format!("Note store has {} events but Notes.txt has {} notes; matched in order", e.len(), labels.len()));
        }
        e.labels = labels.into_iter().chain(std::iter::repeat(String::new())).take(e.len()).collect();
        e.description = "Synapse runtime notes".into();
        return;
    }
    let start = notes.fields.get("Start").and_then(|s| s.split_whitespace().next()).and_then(clock_seconds);
    let Some(start) = start else {
        warnings.push("Notes.txt has notes but no Start time; notes not added as events".into());
        return;
    };
    let onsets: Vec<f64> = notes
        .entries
        .iter()
        .filter_map(|n| clock_seconds(&n.clock))
        .map(|t| if t < start { t + 86_400.0 } else { t } - start)
        .collect();
    if onsets.len() != labels.len() {
        warnings.push("some Notes.txt notes have unreadable times; notes not added as events".into());
        return;
    }
    events.push(EventSeries {
        name: "Note".into(),
        description: "Synapse runtime notes (times from Notes.txt, 1 s resolution)".into(),
        values: (1..=onsets.len()).map(|v| v as f64).collect(),
        onsets,
        offsets: None,
        channels: 1,
        labels,
    });
}

/// Groups a scalar store's per-channel records into events (one row of channel values each).
fn scalars(s: &StoreIndex, block_start: f64, description: Option<String>) -> EventSeries {
    let channels = s.records.iter().map(|r| r.chan as usize).max().unwrap_or(1).max(1);
    let mut series = EventSeries {
        name: s.name.clone(),
        description: description.unwrap_or_else(|| "TDT scalar store".into()),
        channels,
        ..Default::default()
    };
    let mut i = 0;
    while i < s.records.len() {
        let ts = s.records[i].timestamp;
        let mut row = vec![f64::NAN; channels];
        while i < s.records.len() && s.records[i].timestamp == ts {
            let r = &s.records[i];
            let c = (r.chan as usize).clamp(1, channels) - 1;
            row[c] = r.value();
            i += 1;
        }
        series.onsets.push(session_time(ts, block_start));
        series.values.extend(row);
    }
    series
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::tdt::codes;
    use crate::inputs::tdt::tsq::tests::record;
    use crate::inputs::tdt::tsq::TsqIndex;

    #[test]
    fn test_pairs_offsets_by_encoded_name_and_groups_scalars() {
        let mut b = record(10, codes::EVTYPE_MARK, &1u32.to_le_bytes(), 0, 0, 100.0, 0, 0, 0.0);
        for (t, v) in [(101.0, 1.0), (103.0, 2.0)] {
            b.extend(record(10, codes::EVTYPE_STRON, b"MET/", 0, 0, t, f64::to_bits(v), 4, 0.0));
        }
        // Offset store: partner name 'MET/' split over chan ('ME') and sortcode ('T/')
        let chan = u16::from_le_bytes(*b"ME");
        let sort = u16::from_le_bytes(*b"T/");
        b.extend(record(10, codes::EVTYPE_STROFF, b"MET\\", chan, sort, 101.5, 0, 4, 0.0));
        for ch in [2u16, 1] {
            b.extend(record(10, codes::EVTYPE_SCALAR, b"eS1p", ch, 0, 102.0, f64::to_bits(ch as f64 * 10.0), 4, 0.0));
        }
        b.extend(record(10, codes::EVTYPE_MARK, &2u32.to_le_bytes(), 0, 0, 110.0, 0, 0, 0.0));
        let idx = TsqIndex::parse(&b);
        let mut warnings = Vec::new();
        let ev = build(&idx.stores, idx.start, 10.0, &BTreeMap::new(), &mut warnings);

        // Times snap to the 195312.5 Hz device clock (within one tick)
        let close = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1.0 / 195_312.5);
        let met = ev.iter().find(|e| e.name == "MET/").unwrap();
        assert!(close(&met.onsets, &[1.0, 3.0]), "{:?}", met.onsets);
        assert_eq!(met.values, vec![1.0, 2.0]);
        // Second epoc was still open: closed at the block end
        assert!(close(met.offsets.as_deref().unwrap(), &[1.5, 10.0]));
        assert_eq!(warnings.len(), 1);

        let sc = ev.iter().find(|e| e.name == "eS1p").unwrap();
        assert_eq!((sc.channels, sc.values.clone()), (2, vec![10.0, 20.0]));
        assert!(close(&sc.onsets, &[2.0]));
    }
}
