//! Recording file formats for `dsp-kitchen`.
//!
//! Every format implements [`dsp_core::RecordingSource`], so callers read bounded chunks of any
//! recording without knowing how it is stored. [`open`] picks the reader from the path.

mod codec;
pub mod cache;
pub mod mtscomp;
#[cfg(feature = "zarr")]
pub mod nwb;
pub mod raw;
pub mod sources;
pub mod spikeglx;
pub mod synthetic;
#[cfg(feature = "zarr")]
pub mod zarr;

use std::path::Path;

use dsp_core::{DspError, DspResult, RecordingSource};

pub use cache::{cache_path, CacheIdentity, MinMaxCache};
pub use mtscomp::MtscompRecording;
pub use raw::{write_raw, RawParams, RawRecording};
pub use sources::{default_source, open_source, sources, SourceEntry, SourceKind};
pub use spikeglx::SpikeGlxMeta;
pub use synthetic::{SyntheticParams, SyntheticRecording};
#[cfg(feature = "zarr")]
pub use nwb::NwbZarrRecording;
#[cfg(feature = "zarr")]
pub use zarr::{write_zarr, ZarrRecording};

/// Opens a recording, detecting its format from the path:
/// - an NWB file stored as Zarr (`.nwb.zarr`; its largest electrical series),
/// - a Zarr v3 store with a `/traces` array (directory with `zarr.json`, or a `.zarr` path),
/// - SpikeGLX (`.bin` + `key=value` `.meta`), including IBL mtscomp `.cbin` + `.ch`,
/// - a raw binary file with a JSON sidecar (`rec.bin` + `rec.meta`).
pub fn open(path: &Path) -> DspResult<Box<dyn RecordingSource>> {
    if !path.exists() {
        return Err(DspError::Io(format!("{} does not exist", path.display())));
    }
    if path.is_dir() || path.extension().is_some_and(|e| e == "zarr") {
        #[cfg(feature = "zarr")]
        if path.join("zarr.json").exists() {
            if nwb::is_nwb_zarr(path) {
                return Ok(Box::new(NwbZarrRecording::open(path)?));
            }
            if !path.join("traces").join("zarr.json").exists() {
                return Err(DspError::UnsupportedFormat(format!(
                    "{} is a Zarr store but neither an NWB file nor a /traces recording",
                    path.display()
                )));
            }
            return Ok(Box::new(ZarrRecording::open(path)?));
        }
        return Err(DspError::UnsupportedFormat(format!("{} is a folder but not a Zarr v3 store", path.display())));
    }
    if SpikeGlxMeta::is_spikeglx(path) {
        return spikeglx::open(path);
    }
    if path.extension().is_some_and(|e| e == "cbin") {
        return Ok(Box::new(MtscompRecording::open(path)?));
    }
    if RawParams::from_sidecar(path).is_some() {
        return Ok(Box::new(RawRecording::open(path)?));
    }
    Err(DspError::UnsupportedFormat(format!(
        "cannot tell the format of {} (a raw binary file needs a JSON sidecar {})",
        path.display(),
        RawParams::sidecar_path(path).display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::{MemoryOrder, SampleFormat};

    #[test]
    fn test_open_detects_formats() {
        let dir = std::env::temp_dir().join(format!("dsp_io_open_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = SyntheticRecording::new(SyntheticParams { channels: 4, duration_sec: 0.1, ..Default::default() }).unwrap();

        let bin = dir.join("rec.bin");
        write_raw(&src, &bin, SampleFormat::I16, MemoryOrder::TimeMajor, 0.25, 1000, |_, _| {}).unwrap();
        let rec = open(&bin).unwrap();
        assert_eq!(rec.info().format, SampleFormat::I16);
        assert_eq!(rec.info().samples, src.info().samples);

        #[cfg(feature = "zarr")]
        {
            let z = dir.join("rec.zarr");
            write_zarr(&src, &z, 1000, |_, _| {}).unwrap();
            assert_eq!(open(&z).unwrap().info().channel_count(), 4);
        }

        std::fs::write(dir.join("other.dat"), [0u8; 8]).unwrap();
        assert!(matches!(open(&dir.join("other.dat")), Err(DspError::UnsupportedFormat(_))));
        assert!(open(&dir.join("missing.bin")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `read_stored` decoded with each channel's gain and offset equals `read`.
    pub(crate) fn assert_stored_matches(rec: &dyn RecordingSource, channels: &[usize], samples: std::ops::Range<u64>) {
        let info = rec.info();
        let n = (samples.end - samples.start) as usize;
        let mut values = vec![0.0f32; channels.len() * n];
        rec.read(channels, samples.clone(), &mut values).unwrap();
        let mut bytes = vec![0u8; channels.len() * n * info.format.bytes()];
        rec.read_stored(channels, samples, &mut bytes).unwrap();
        let mut decoded = vec![0.0f32; values.len()];
        for (i, &c) in channels.iter().enumerate() {
            let ch = &info.channels[c];
            let b = info.format.bytes();
            crate::codec::decode_run(info.format, &bytes[i * n * b..(i + 1) * n * b], &mut decoded[i * n..(i + 1) * n], ch.gain_uv, ch.offset_uv);
        }
        assert_eq!(decoded, values, "{}", info.name);
    }

    #[test]
    fn stored_reads_match_scaled_reads() {
        let dir = std::env::temp_dir().join(format!("dsp_io_stored_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = SyntheticRecording::new(SyntheticParams { channels: 5, duration_sec: 0.1, ..Default::default() }).unwrap();
        let channels = [4usize, 0, 2];

        for (name, format, order, gain) in [
            ("tm_i16.bin", SampleFormat::I16, MemoryOrder::TimeMajor, 0.25),
            ("cm_i16.bin", SampleFormat::I16, MemoryOrder::ChannelMajor, 0.5),
            ("cm_f32.bin", SampleFormat::F32, MemoryOrder::ChannelMajor, 1.0),
            ("tm_i32.bin", SampleFormat::I32, MemoryOrder::TimeMajor, 0.01),
        ] {
            let bin = dir.join(name);
            write_raw(&src, &bin, format, order, gain, 700, |_, _| {}).unwrap();
            let rec = open(&bin).unwrap();
            assert_stored_matches(rec.as_ref(), &channels, 100..1_900);
            // A sliced view maps channels and samples onto its parent
            let parent: std::sync::Arc<dyn RecordingSource> = std::sync::Arc::from(rec);
            let sliced = dsp_core::SlicedRecording::new(parent, 50..2_000, Some(vec![3, 1, 4])).unwrap();
            assert_stored_matches(&sliced, &[2, 0], 10..900);
        }

        // Synthetic f32 µV with unit gain: the default implementation
        assert_stored_matches(&src, &channels, 0..500);

        #[cfg(feature = "zarr")]
        {
            let z = dir.join("rec.zarr");
            write_zarr(&src, &z, 700, |_, _| {}).unwrap();
            assert_stored_matches(open(&z).unwrap().as_ref(), &channels, 100..1_900);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
