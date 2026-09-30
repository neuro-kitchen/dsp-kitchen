//! The signals ("sources") a file offers, listed cheaply and opened on demand.
//!
//! An NWB store holds many series (HD-EMG, EMG, temperature, …) and a SpikeGLX run several
//! streams (ap, lf, nidq); plain recordings are one source. Listing reads only metadata.

use std::path::{Path, PathBuf};

use dsp_core::{DspError, DspResult, RecordingSource};

use crate::spikeglx::SpikeGlxMeta;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// Voltage from electrodes (values in µV).
    Electrical,
    /// Anything else (pressure, temperature, analog inputs, …).
    Other,
}

/// One openable signal of a file.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceEntry {
    /// Stable id within the file (NWB series path, SpikeGLX file name, `main`).
    pub id: String,
    pub name: String,
    pub kind: SourceKind,
    pub channels: usize,
    pub samples: u64,
    pub sample_rate: f64,
    pub start_time_sec: f64,
    /// Unit of the values a read returns (`µV` for electrical sources).
    pub unit: String,
}

impl SourceEntry {
    pub fn duration_sec(&self) -> f64 {
        if self.sample_rate > 0.0 { self.samples as f64 / self.sample_rate } else { 0.0 }
    }

    /// e.g. `HDEMG · 32 ch · 24.4 kHz · µV`
    pub fn summary(&self) -> String {
        let rate = if self.sample_rate >= 1000.0 { format!("{:.1} kHz", self.sample_rate / 1000.0) } else { format!("{:.1} Hz", self.sample_rate) };
        format!("{} · {} ch · {rate} · {}", self.name, self.channels, self.unit)
    }
}

/// The id [`sources`] gives a single-recording file.
pub const MAIN: &str = "main";

/// Every source of `path`, cheapest metadata only.
pub fn sources(path: &Path) -> DspResult<Vec<SourceEntry>> {
    #[cfg(feature = "zarr")]
    if crate::nwb::is_nwb_zarr(path) {
        let list: Vec<SourceEntry> = crate::nwb::list_series(path)
            .into_iter()
            .map(|s| SourceEntry {
                name: s.path.rsplit('/').next().unwrap_or(&s.path).to_string(),
                kind: if s.neurodata_type == "ElectricalSeries" { SourceKind::Electrical } else { SourceKind::Other },
                channels: s.channels,
                samples: s.samples,
                sample_rate: s.rate,
                start_time_sec: s.start_time,
                unit: if s.neurodata_type == "ElectricalSeries" || s.unit == "volts" { "µV".into() } else { s.unit.clone() },
                id: s.path,
            })
            .collect();
        if list.is_empty() {
            return Err(DspError::UnsupportedFormat(format!("{} has no continuous series", path.display())));
        }
        return Ok(list);
    }
    if path.is_file() && SpikeGlxMeta::is_spikeglx(path) {
        return Ok(spikeglx_streams(path).iter().filter_map(|p| spikeglx_entry(p)).collect());
    }
    let rec = crate::open(path)?;
    let info = rec.info();
    Ok(vec![SourceEntry {
        id: MAIN.into(),
        name: info.name.clone(),
        kind: SourceKind::Electrical,
        channels: info.channel_count(),
        samples: info.samples,
        sample_rate: info.sample_rate_hz(),
        start_time_sec: info.start_time_sec,
        unit: info.metadata.get("unit").map_or_else(|| "µV".into(), |u| u.replace("uV", "µV")),
    }])
}

/// Opens source `id` of `path` (an id from [`sources`]).
pub fn open_source(path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
    #[cfg(feature = "zarr")]
    if crate::nwb::is_nwb_zarr(path) {
        return Ok(Box::new(crate::nwb::NwbZarrRecording::open_series(path, id)?));
    }
    if path.is_file() && SpikeGlxMeta::is_spikeglx(path) {
        let file = spikeglx_streams(path).into_iter().find(|p| p.file_name().is_some_and(|n| n.to_string_lossy() == id));
        return match file {
            Some(f) => crate::spikeglx::open(&f),
            None => Err(DspError::InvalidConfig(format!("no SpikeGLX stream {id} next to {}", path.display()))),
        };
    }
    if id == MAIN {
        return crate::open(path);
    }
    Err(DspError::InvalidConfig(format!("{} has no source {id}", path.display())))
}

/// SpikeGLX streams of the same run in the same folder: `<run>.imecN.ap.bin`, `.lf.bin`,
/// `<run>.nidq.bin` (and `.cbin`), sorted by name.
fn spikeglx_streams(path: &Path) -> Vec<PathBuf> {
    let name = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let run = name.split(".imec").next().unwrap_or(&name).split(".nidq").next().unwrap_or(&name).to_string();
    let Some(dir) = path.parent() else { return vec![path.to_path_buf()] };
    let mut out: Vec<PathBuf> = std::fs::read_dir(if dir.as_os_str().is_empty() { Path::new(".") } else { dir })
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let n = p.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            n.starts_with(&run) && (n.ends_with(".bin") || n.ends_with(".cbin")) && SpikeGlxMeta::is_spikeglx(p)
        })
        .collect();
    out.sort();
    if out.is_empty() {
        out.push(path.to_path_buf());
    }
    out
}

fn spikeglx_entry(path: &Path) -> Option<SourceEntry> {
    let meta = SpikeGlxMeta::read(&SpikeGlxMeta::path_for(path)).ok()?;
    let channels: usize = meta.get("nSavedChans")?.parse().ok()?;
    let rate: f64 = meta.get("imSampRate").or_else(|| meta.get("niSampRate"))?.parse().ok()?;
    let file = path.file_name()?.to_string_lossy().into_owned();
    // Payload size: the file itself for .bin (may still be growing), the meta for .cbin
    let bytes = if file.ends_with(".cbin") { meta.get("fileSizeBytes")?.parse().ok()? } else { std::fs::metadata(path).ok()?.len() };
    let nidq = meta.get("typeThis") == Some("nidq");
    Some(SourceEntry {
        name: stream_name(&file),
        kind: if nidq { SourceKind::Other } else { SourceKind::Electrical },
        channels,
        samples: bytes / (2 * channels.max(1) as u64),
        sample_rate: rate,
        start_time_sec: 0.0,
        unit: "µV".into(),
        id: file,
    })
}

/// `run_g0_t0.imec0.ap.bin` → `imec0.ap`; `run_g0_t0.nidq.bin` → `nidq`.
fn stream_name(file: &str) -> String {
    let stem = file.trim_end_matches(".cbin").trim_end_matches(".bin");
    let at = stem.find(".imec").or_else(|| stem.find(".nidq")).or_else(|| stem.find(".obx"));
    at.map_or_else(|| stem.to_string(), |i| stem[i + 1..].to_string())
}

/// The source to show first: the largest electrical source, else the largest one.
pub fn default_source(list: &[SourceEntry]) -> Option<&SourceEntry> {
    // Most data first; equal sizes: the higher sample rate (e.g. SpikeGLX ap over lf)
    let size = |s: &&SourceEntry| (s.samples * s.channels as u64, s.sample_rate as u64);
    list.iter().filter(|s| s.kind == SourceKind::Electrical).max_by_key(size).or_else(|| list.iter().max_by_key(size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::{MemoryOrder, SampleFormat};

    #[test]
    fn test_single_recording_is_one_source() {
        let dir = std::env::temp_dir().join(format!("dsp_io_src_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = crate::SyntheticRecording::new(crate::SyntheticParams { channels: 3, duration_sec: 0.1, ..Default::default() }).unwrap();
        let bin = dir.join("rec.bin");
        crate::write_raw(&src, &bin, SampleFormat::F32, MemoryOrder::ChannelMajor, 1.0, 1000, |_, _| {}).unwrap();

        let list = sources(&bin).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!((list[0].id.as_str(), list[0].channels, list[0].unit.as_str()), (MAIN, 3, "µV"));
        assert_eq!(open_source(&bin, MAIN).unwrap().info().channel_count(), 3);
        assert!(open_source(&bin, "nope").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_spikeglx_run_streams() {
        let dir = std::env::temp_dir().join(format!("dsp_io_sglx_run_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let meta = |ch: usize, rate: &str, ty: &str| {
            format!("nSavedChans={ch}\n{rate}\ntypeThis={ty}\nimAiRangeMax=0.6\nniAiRangeMax=5\nsnsApLfSy={},0,1\n", ch - 1)
        };
        for (file, ch, rate, ty) in [
            ("run_g0_t0.imec0.ap.bin", 3, "imSampRate=30000", "imec"),
            ("run_g0_t0.imec0.lf.bin", 3, "imSampRate=2500", "imec"),
            ("run_g0_t0.nidq.bin", 2, "niSampRate=25000", "nidq"),
        ] {
            std::fs::write(dir.join(file), vec![0u8; ch * 2 * 10]).unwrap();
            std::fs::write(SpikeGlxMeta::path_for(&dir.join(file)), meta(ch, rate, ty)).unwrap();
        }
        let list = sources(&dir.join("run_g0_t0.imec0.ap.bin")).unwrap();
        let names: Vec<(&str, SourceKind, f64)> = list.iter().map(|s| (s.name.as_str(), s.kind, s.sample_rate)).collect();
        assert_eq!(
            names,
            vec![("imec0.ap", SourceKind::Electrical, 30000.0), ("imec0.lf", SourceKind::Electrical, 2500.0), ("nidq", SourceKind::Other, 25000.0)]
        );
        assert_eq!(list[0].samples, 10);
        assert_eq!(default_source(&list).unwrap().name, "imec0.ap");
        let lf = open_source(&dir.join("run_g0_t0.imec0.ap.bin"), "run_g0_t0.imec0.lf.bin").unwrap();
        assert_eq!(lf.info().sample_rate_hz(), 2500.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
