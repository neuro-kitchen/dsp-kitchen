//! SpikeGLX recordings: `*.imecN.ap.bin` / `*.lf.bin` / `*.nidq.bin` with a `key=value` `.meta`.
//!
//! Samples are int16, interleaved. Conversion to µV follows the SpikeGLX metadata guide:
//! `µV = i * AiRangeMax / MaxInt / gain * 1e6`, with per-channel gains from `imroTbl` (imec)
//! or `niMNGain` / `niMAGain` (nidq). Probe geometry comes from `snsGeomMap` when present,
//! otherwise from `snsShankMap` and the probe type's electrode pitch. Compressed IBL files
//! (`.cbin` + `.ch`) are read through [`MtscompRecording`](crate::mtscomp::MtscompRecording).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dsp_core::layout::{Position3D, SensorLayout, SensorSite};
use dsp_core::{DspError, DspResult, MemoryOrder, RecordingInfo, RecordingSource, SampleFormat};

use crate::mtscomp::MtscompRecording;
use crate::raw::{RawParams, RawRecording};

/// Parsed `.meta` file (`~` prefixes of table keys are dropped).
#[derive(Debug, Clone, Default)]
pub struct SpikeGlxMeta(pub BTreeMap<String, String>);

impl SpikeGlxMeta {
    pub fn path_for(data: &Path) -> PathBuf {
        data.with_extension("meta")
    }

    pub fn read(path: &Path) -> DspResult<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
        Ok(Self::parse(&text))
    }

    pub fn parse(text: &str) -> Self {
        let map = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.trim().trim_start_matches('~').to_string(), v.trim().to_string()))
            .collect();
        Self(map)
    }

    /// True when `data`'s `.meta` sidecar is a SpikeGLX file.
    pub fn is_spikeglx(data: &Path) -> bool {
        std::fs::read_to_string(Self::path_for(data)).is_ok_and(|t| t.lines().any(|l| l.starts_with("nSavedChans=")))
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    fn f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.parse().ok()
    }

    fn require_f64(&self, key: &str) -> DspResult<f64> {
        self.f64(key).ok_or_else(|| DspError::InvalidConfig(format!("SpikeGLX meta is missing {key}")))
    }

    fn counts(&self, key: &str) -> Option<Vec<usize>> {
        self.get(key)?.split(',').map(|v| v.trim().parse().ok()).collect()
    }

    fn is_nidq(&self) -> bool {
        self.get("typeThis") == Some("nidq")
    }

    /// Acquired channel id of each saved channel, in file order.
    fn saved_channels(&self, n_saved: usize) -> Vec<usize> {
        match self.get("snsSaveChanSubset") {
            None | Some("all") => (0..n_saved).collect(),
            Some(list) => list
                .split(',')
                .flat_map(|part| match part.split_once(':') {
                    Some((a, b)) => a.trim().parse().unwrap_or(0)..b.trim().parse::<usize>().unwrap_or(0) + 1,
                    None => {
                        let v = part.trim().parse().unwrap_or(0);
                        v..v + 1
                    }
                })
                .collect(),
        }
    }

    /// Probe type: `imDatPrb_type`, else 0 (1.0 / 3B) — 3A probes have neither and act like 1.0.
    fn probe_type(&self) -> u32 {
        self.get("imDatPrb_type").and_then(|v| v.parse().ok()).unwrap_or(0)
    }
}

/// Neuropixels 2.0 families (fixed gain 80, 15 µm rows, 32 µm columns, 250 µm between shanks).
fn is_np2(probe_type: u32) -> bool {
    matches!(probe_type, 21 | 24 | 2003 | 2004 | 2013 | 2014 | 2020 | 2021)
}

/// Splits `(a b c)(d e f)` tables into groups of fields.
fn table(s: &str) -> Vec<Vec<&str>> {
    s.trim_start_matches('(')
        .trim_end_matches(')')
        .split(")(")
        .map(|g| g.split(|c: char| c == ' ' || c == ',' || c == ':').filter(|f| !f.is_empty()).collect())
        .collect()
}

/// `(ap_gain, lf_gain)` per acquired AP channel, from `imroTbl`.
fn imro_gains(meta: &SpikeGlxMeta, n_ap: usize) -> Vec<(f32, f32)> {
    let default = if is_np2(meta.probe_type()) { (80.0, 80.0) } else { (500.0, 250.0) };
    let mut gains = vec![default; n_ap];
    let Some(imro) = meta.get("imroTbl") else { return gains };
    let groups = table(imro);
    // Header is (type,n) or (serial,type,n) for 3A probes
    let header_type = groups.first().and_then(|h| h.get(h.len().saturating_sub(2))).and_then(|v| v.parse::<u32>().ok());
    if header_type.is_some_and(is_np2) || is_np2(meta.probe_type()) {
        return gains;
    }
    for (i, e) in groups.iter().skip(1).take(n_ap).enumerate() {
        let ap = e.get(3).and_then(|v| v.parse().ok()).unwrap_or(default.0);
        let lf = e.get(4).and_then(|v| v.parse().ok()).unwrap_or(default.1);
        gains[i] = (ap, lf);
    }
    gains
}

/// Positions (µm) and shank of each saved neural channel, in file order.
fn geometry(meta: &SpikeGlxMeta) -> Option<Vec<(f32, f32, usize)>> {
    if let Some(geom) = meta.get("snsGeomMap") {
        // (header)(shank:x:z:use)... — x/z already in µm; shanks are `shank_pitch` apart
        let groups = table(geom);
        let shank_pitch: f32 = groups.first()?.get(2).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        return groups
            .iter()
            .skip(1)
            .map(|e| {
                let shank: usize = e.first()?.parse().ok()?;
                let x: f32 = e.get(1)?.parse().ok()?;
                let z: f32 = e.get(2)?.parse().ok()?;
                Some((x + shank as f32 * shank_pitch, z, shank))
            })
            .collect();
    }
    let shank_map = meta.get("snsShankMap")?;
    let np2 = is_np2(meta.probe_type());
    table(shank_map)
        .iter()
        .skip(1)
        .map(|e| {
            let shank: usize = e.first()?.parse().ok()?;
            let col: usize = e.get(1)?.parse().ok()?;
            let row: usize = e.get(2)?.parse().ok()?;
            Some(if np2 {
                (shank as f32 * 250.0 + col as f32 * 32.0, row as f32 * 15.0, shank)
            } else {
                // 1.0 staggered checkerboard: 43/11 µm on even rows, 59/27 µm on odd rows
                const X: [f32; 4] = [43.0, 11.0, 59.0, 27.0];
                (X[(row % 2) * 2 + col.min(1)], row as f32 * 20.0, shank)
            })
        })
        .collect()
}

/// Applies SpikeGLX names, µV gains, geometry and metadata to `info` (samples unchanged).
pub fn apply_meta(meta: &SpikeGlxMeta, mut info: RecordingInfo, lf_stream: bool) -> DspResult<RecordingInfo> {
    let n_saved = info.channels.len();
    let saved = meta.saved_channels(n_saved);
    if saved.len() != n_saved {
        return Err(DspError::InvalidConfig(format!("snsSaveChanSubset lists {} channels, file has {n_saved}", saved.len())));
    }

    let mut sync = Vec::new();
    if meta.is_nidq() {
        // Acquired order: MN, MA, XA analog, then digital words
        let c = meta.counts("acqMnMaXaDw").or_else(|| meta.counts("snsMnMaXaDw")).unwrap_or_else(|| vec![0, 0, n_saved, 0]);
        let (mn, ma, xa) = (c[0], c[1], c[2]);
        let range = meta.require_f64("niAiRangeMax")?;
        let max_int = meta.f64("niMaxInt").unwrap_or(32768.0);
        let (g_mn, g_ma) = (meta.f64("niMNGain").unwrap_or(1.0), meta.f64("niMAGain").unwrap_or(1.0));
        for (ch, &id) in info.channels.iter_mut().zip(&saved) {
            let (name, gain) = if id < mn {
                (format!("MN{id}"), Some(g_mn))
            } else if id < mn + ma {
                (format!("MA{}", id - mn), Some(g_ma))
            } else if id < mn + ma + xa {
                (format!("XA{}", id - mn - ma), Some(1.0))
            } else {
                (format!("DW{}", id - mn - ma - xa), None)
            };
            ch.name = name;
            ch.gain_uv = gain.map_or(1.0, |g| (range / max_int / g * 1e6) as f32);
        }
    } else {
        let c = meta.counts("acqApLfSy").or_else(|| meta.counts("snsApLfSy")).unwrap_or_else(|| vec![n_saved - 1, 0, 1]);
        let (n_ap, n_lf) = (c[0], c[1]);
        let range = meta.require_f64("imAiRangeMax")?;
        let max_int = meta.f64("imMaxInt").unwrap_or(512.0);
        let gains = imro_gains(meta, n_ap);
        for (i, (ch, &id)) in info.channels.iter_mut().zip(&saved).enumerate() {
            let (name, gain) = if id < n_ap {
                (format!("AP{id}"), Some(gains[id].0))
            } else if id < n_ap + n_lf {
                (format!("LF{}", id - n_ap), Some(gains[(id - n_ap) % n_ap.max(1)].1))
            } else {
                sync.push(i);
                (format!("SY{}", id - n_ap - n_lf), None)
            };
            // An LF-only file's LF channels carry the LF gain even when numbered like AP
            let gain = if lf_stream && id < n_ap { Some(gains[id].1) } else { gain };
            ch.name = name;
            ch.gain_uv = gain.map_or(1.0, |g| (range / max_int / g as f64 * 1e6) as f32);
        }

        if let Some(geom) = geometry(meta) {
            let neural: Vec<usize> = (0..n_saved).filter(|i| !sync.contains(i)).collect();
            let contacts = neural
                .iter()
                .zip(&geom)
                .map(|(&file_ch, &(x, y, shank))| SensorSite::new(file_ch, Position3D::new(x, y, 0.0), shank))
                .collect();
            let probe = meta.get("imDatPrb_pn").unwrap_or(if meta.get("imProbeOpt").is_some() { "3A" } else { "Neuropixels" });
            info.layout = Some(SensorLayout::new(probe, contacts));
        }
    }

    for key in ["typeThis", "imDatPrb_type", "imDatPrb_pn", "imDatPrb_sn", "imProbeOpt", "appVersion", "fileCreateTime", "firstSample"] {
        if let Some(v) = meta.get(key) {
            info.metadata.insert(key.to_string(), v.to_string());
        }
    }
    if !sync.is_empty() {
        info.metadata.insert("sync_channels".into(), sync.iter().map(usize::to_string).collect::<Vec<_>>().join(","));
    }
    info.metadata.insert("format".into(), "SpikeGLX".into());
    Ok(info)
}

/// Opens a SpikeGLX `.bin` (memory-mapped) or IBL `.cbin` (mtscomp) with its `.meta`.
pub fn open(path: &Path) -> DspResult<Box<dyn RecordingSource>> {
    let meta = SpikeGlxMeta::read(&SpikeGlxMeta::path_for(path))?;
    let n_saved = meta.require_f64("nSavedChans")? as usize;
    let rate = meta.f64("imSampRate").or_else(|| meta.f64("niSampRate")).ok_or_else(|| DspError::InvalidConfig("SpikeGLX meta has no sample rate".into()))?;
    let lf = path.to_string_lossy().contains(".lf.");

    if path.extension().is_some_and(|e| e == "cbin") {
        let rec = MtscompRecording::open(path)?;
        if rec.info().channel_count() != n_saved {
            return Err(DspError::InvalidConfig(format!("{} has {} channels, meta says {n_saved}", path.display(), rec.info().channel_count())));
        }
        let info = apply_meta(&meta, rec.info().clone(), lf)?;
        return Ok(Box::new(rec.with_info(info)));
    }

    let mut params = RawParams::new(n_saved, rate, SampleFormat::I16, MemoryOrder::TimeMajor);
    // Trust the file size over fileSizeBytes: a recording can be read while still being written
    params.samples = None;
    let rec = RawRecording::open_with(path, &params)?;
    let info = apply_meta(&meta, rec.info().clone(), lf)?;
    Ok(Box::new(rec.with_info(info)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::SampleRate;

    const META_3A: &str = "nSavedChans=5\nimSampRate=30000\nimAiRangeMax=0.6\ntypeThis=imec\nimProbeOpt=3\n\
acqApLfSy=4,4,1\nsnsApLfSy=4,0,1\nsnsSaveChanSubset=0:3,8\n\
~imroTbl=(641251510,3,4)(0 0 0 500 250)(1 0 0 500 250)(2 0 0 1000 250)(3 0 0 500 250)\n\
~snsShankMap=(1,2,480)(0:0:0:1)(0:1:0:1)(0:0:1:1)(0:1:1:1)\n";

    fn info(n: usize) -> RecordingInfo {
        RecordingInfo::new("t", n, 10, SampleRate::new(30000.0).unwrap(), SampleFormat::I16, MemoryOrder::TimeMajor)
    }

    #[test]
    fn test_imec_3a_gains_names_geometry() {
        let meta = SpikeGlxMeta::parse(META_3A);
        let i = apply_meta(&meta, info(5), false).unwrap();
        let names: Vec<&str> = i.channels.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["AP0", "AP1", "AP2", "AP3", "SY0"]);
        // 0.6 V / 512 / 500 = 2.34375 µV per bit; channel 2 has gain 1000
        assert!((i.channels[0].gain_uv - 2.34375).abs() < 1e-5);
        assert!((i.channels[2].gain_uv - 1.171875).abs() < 1e-5);
        assert_eq!(i.channels[4].gain_uv, 1.0);
        assert_eq!(i.metadata["sync_channels"], "4");

        let layout = i.layout.unwrap();
        assert_eq!(layout.total_channels(), 4);
        let p: Vec<(f32, f32)> = layout.sites().iter().map(|s| (s.position.x_um, s.position.y_um)).collect();
        assert_eq!(p, [(43.0, 0.0), (11.0, 0.0), (59.0, 20.0), (27.0, 20.0)]);
    }

    #[test]
    fn test_np2_fixed_gain_and_geom_map() {
        let meta = SpikeGlxMeta::parse(
            "nSavedChans=3\nimSampRate=30000\nimAiRangeMax=0.5\nimMaxInt=8192\ntypeThis=imec\nimDatPrb_type=24\n\
snsApLfSy=2,0,1\n~imroTbl=(24,2)(0 0 0 0 0)(1 0 0 0 1)\n~snsGeomMap=(NP2014,4,250,70)(0:27:0:1)(1:59:15:1)\n",
        );
        let i = apply_meta(&meta, info(3), false).unwrap();
        // 0.5 V / 8192 / 80 = 0.762939 µV per bit
        assert!((i.channels[1].gain_uv - 0.762_939).abs() < 1e-5);
        let p: Vec<(f32, f32, usize)> = i.layout.unwrap().sites().iter().map(|s| (s.position.x_um, s.position.y_um, s.shank_id)).collect();
        assert_eq!(p, [(27.0, 0.0, 0), (309.0, 15.0, 1)]);
    }

    #[test]
    fn test_open_bin_with_meta() {
        let dir = std::env::temp_dir().join(format!("dsp_io_sglx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("rec_g0_t0.imec0.ap.bin");
        // 2 samples x 5 channels interleaved
        let bytes: Vec<u8> = (0..10i16).flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&bin, bytes).unwrap();
        std::fs::write(SpikeGlxMeta::path_for(&bin), META_3A).unwrap();

        assert!(SpikeGlxMeta::is_spikeglx(&bin));
        let rec = crate::open(&bin).unwrap();
        assert_eq!(rec.info().samples, 2);
        let mut out = [0.0; 2];
        rec.read(&[4], 0..2, &mut out).unwrap();
        assert_eq!(out, [4.0, 9.0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
