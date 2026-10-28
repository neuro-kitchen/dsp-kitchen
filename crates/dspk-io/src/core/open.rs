//! Format detection: [`open`] picks the reader from the path.

use std::path::Path;

use dspk_core::{DspError, DspResult, RecordingSource};

use super::format::Format;
use super::sources::SourceEntry;
use crate::registry::formats;

/// The format of `path`: the first compiled-in [`Format`] that detects it.
pub fn detect(path: &Path) -> DspResult<&'static dyn Format> {
    detect_in(formats(), path)
}

/// The format of `path`: the first of `formats` that detects it. For crates that add formats to
/// the generic ones (e.g. `dspk-synapse-io`).
pub fn detect_in(formats: &[&'static dyn Format], path: &Path) -> DspResult<&'static dyn Format> {
    if !path.exists() {
        return Err(DspError::Io(format!("{} does not exist", path.display())));
    }
    formats.iter().copied().find(|f| f.detect(path)).ok_or_else(|| {
        let names: Vec<&str> = formats.iter().map(|f| f.name()).collect();
        DspError::UnsupportedFormat(format!(
            "cannot tell the format of {} (compiled-in formats: {}; a raw binary file needs a JSON sidecar)",
            path.display(),
            names.join(", ")
        ))
    })
}

/// Opens the default source of `path` (see [`Format::open_default`]).
pub fn open(path: &Path) -> DspResult<Box<dyn RecordingSource>> {
    detect(path)?.open_default(path)
}

/// Every source of `path`, from metadata only.
pub fn sources(path: &Path) -> DspResult<Vec<SourceEntry>> {
    detect(path)?.sources(path)
}

/// Opens source `id` of `path` (an id from [`sources`]).
pub fn open_source(path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
    detect(path)?.open(path, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_core::{MemoryOrder, SampleFormat};
    #[cfg(feature = "zarr")]
    use crate::core::CachedRecording;
    use crate::generic::raw::write_raw;
    #[cfg(feature = "zarr")]
    use crate::generic::zarr_traces::write_zarr;
    use crate::test_support::{assert_native_matches, assert_stored_matches};
    use dspk_core::MemoryRecording;

    /// `channels` × 3 000 samples at 30 kHz: a different sine per channel, in µV.
    fn source(channels: usize) -> MemoryRecording {
        const SAMPLES: usize = 3_000;
        let data = (0..channels)
            .flat_map(|c| (0..SAMPLES).map(move |t| 100.0 * ((t as f32) * 0.01 * (c + 1) as f32).sin()))
            .collect();
        MemoryRecording::new("source", data, channels, 30_000.0).unwrap()
    }

    #[test]
    fn test_open_detects_formats() {
        let dir = std::env::temp_dir().join(format!("dspk_io_open_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = source(4);

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

    #[test]
    fn test_single_recording_is_one_source() {
        let dir = std::env::temp_dir().join(format!("dspk_io_src_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = source(3);
        let bin = dir.join("rec.bin");
        write_raw(&src, &bin, SampleFormat::F32, MemoryOrder::ChannelMajor, 1.0, 1000, |_, _| {}).unwrap();

        let list = sources(&bin).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!((list[0].id.as_str(), list[0].channels), (crate::core::sources::MAIN, 3));
        // The sidecar keeps the source's unit
        assert_eq!(list[0].unit, src.info().channels[0].unit);
        assert_eq!(open_source(&bin, crate::core::sources::MAIN).unwrap().info().channel_count(), 3);
        assert!(open_source(&bin, "nope").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stored_reads_match_scaled_reads() {
        let dir = std::env::temp_dir().join(format!("dspk_io_stored_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = source(5);
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
            assert_native_matches(rec.as_ref(), 100..1_900);
            // A sliced view maps channels and samples onto its parent
            let parent: std::sync::Arc<dyn RecordingSource> = std::sync::Arc::from(rec);
            let sliced = dspk_core::SlicedRecording::new(parent, 50..2_000, Some(vec![3, 1, 4])).unwrap();
            assert_stored_matches(&sliced, &[2, 0], 10..900);
        }

        // In-memory f32 µV with unit gain: the default implementation
        assert_stored_matches(&src, &channels, 0..500);

        #[cfg(feature = "zarr")]
        {
            let z = dir.join("rec.zarr");
            write_zarr(&src, &z, 700, |_, _| {}).unwrap();
            assert_stored_matches(open(&z).unwrap().as_ref(), &channels, 100..1_900);
            assert_native_matches(open(&z).unwrap().as_ref(), 100..1_900);
            let cached = CachedRecording::wrap(open(&z).unwrap(), 1 << 20);
            assert_native_matches(cached.as_ref(), 100..1_900);
            assert_stored_matches(cached.as_ref(), &channels, 100..1_900);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
