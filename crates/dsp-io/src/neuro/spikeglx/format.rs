//! [`Format`] for SpikeGLX runs: one source per stream of the run (`imecN.ap`, `imecN.lf`,
//! `nidq`), each a `.bin` or `.cbin` with its `.meta`.

use std::path::{Path, PathBuf};

use dsp_core::{DspError, DspResult, RationalTime, RecordingSource, SampleFormat, SampleRate, SignalUnit};

use super::{probe_layout, SpikeGlxMeta};
use crate::core::sources::{SourceEntry, SourceKind};
use crate::core::Format;
use crate::neuro::probe::{ProbeSource, SensorLayout};

pub struct SpikeGlx;

impl Format for SpikeGlx {
    fn name(&self) -> &'static str {
        "spikeglx"
    }

    fn detect(&self, path: &Path) -> bool {
        path.is_file() && SpikeGlxMeta::is_spikeglx(path)
    }

    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>> {
        Ok(streams(path).iter().filter_map(|p| entry(p)).collect())
    }

    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
        super::open(&stream(path, id)?)
    }

    /// The stream `path` names.
    fn open_default(&self, path: &Path) -> DspResult<Box<dyn RecordingSource>> {
        super::open(path)
    }
}

impl ProbeSource for SpikeGlx {
    fn probe(&self, path: &Path, id: &str) -> DspResult<Option<SensorLayout>> {
        let meta = SpikeGlxMeta::read(&SpikeGlxMeta::path_for(&stream(path, id)?))?;
        Ok(probe_layout(&meta))
    }
}

/// Stream `id` (a file name from [`streams`]) of the run `path` belongs to.
fn stream(path: &Path, id: &str) -> DspResult<PathBuf> {
    streams(path)
        .into_iter()
        .find(|p| p.file_name().is_some_and(|n| n.to_string_lossy() == id))
        .ok_or_else(|| DspError::InvalidConfig(format!("no SpikeGLX stream {id} next to {}", path.display())))
}

/// SpikeGLX streams of the same run in the same folder: `<run>.imecN.ap.bin`, `.lf.bin`,
/// `<run>.nidq.bin` (and `.cbin`), sorted by name.
fn streams(path: &Path) -> Vec<PathBuf> {
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

fn entry(path: &Path) -> Option<SourceEntry> {
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
        sample_rate: SampleRate::new(rate).ok()?,
        start_time: RationalTime::ZERO,
        format: SampleFormat::I16,
        unit: SignalUnit::Microvolt,
        id: file,
    })
}

/// `run_g0_t0.imec0.ap.bin` → `imec0.ap`; `run_g0_t0.nidq.bin` → `nidq`.
fn stream_name(file: &str) -> String {
    let stem = file.trim_end_matches(".cbin").trim_end_matches(".bin");
    let at = stem.find(".imec").or_else(|| stem.find(".nidq")).or_else(|| stem.find(".obx"));
    at.map_or_else(|| stem.to_string(), |i| stem[i + 1..].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::default_source;

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
        let list = crate::sources(&dir.join("run_g0_t0.imec0.ap.bin")).unwrap();
        let names: Vec<(&str, SourceKind, f64)> = list.iter().map(|s| (s.name.as_str(), s.kind, s.sample_rate.rate_hz())).collect();
        assert_eq!(
            names,
            vec![("imec0.ap", SourceKind::Electrical, 30000.0), ("imec0.lf", SourceKind::Electrical, 2500.0), ("nidq", SourceKind::Other, 25000.0)]
        );
        assert_eq!(list[0].samples, 10);
        assert_eq!(default_source(&list).unwrap().name, "imec0.ap");
        let lf = crate::open_source(&dir.join("run_g0_t0.imec0.ap.bin"), "run_g0_t0.imec0.lf.bin").unwrap();
        assert_eq!(lf.info().sample_rate_hz(), 2500.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
