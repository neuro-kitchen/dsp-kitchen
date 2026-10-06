//! Format detection: [`open`] picks the reader from the path.

use std::path::Path;

use dsp_core::{DspError, DspResult, RecordingSource};

use super::format::Format;
use super::sources::SourceEntry;
use crate::registry::formats;

/// The format of `path`: the first compiled-in [`Format`] that detects it.
pub fn detect(path: &Path) -> DspResult<&'static dyn Format> {
    if !path.exists() {
        return Err(DspError::Io(format!("{} does not exist", path.display())));
    }
    formats().iter().copied().find(|f| f.detect(path)).ok_or_else(|| {
        let names: Vec<&str> = formats().iter().map(|f| f.name()).collect();
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

#[cfg(all(test, feature = "neuro"))]
pub(crate) mod tests {
    use super::*;
    use dsp_core::{MemoryOrder, SampleFormat};
    use crate::core::CachedRecording;
    use crate::generic::raw::write_raw;
    #[cfg(feature = "zarr")]
    use crate::generic::zarr_traces::write_zarr;
    use crate::neuro::synthetic::{SyntheticParams, SyntheticRecording};

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

    #[test]
    fn test_single_recording_is_one_source() {
        let dir = std::env::temp_dir().join(format!("dsp_io_src_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let src = SyntheticRecording::new(SyntheticParams { channels: 3, duration_sec: 0.1, ..Default::default() }).unwrap();
        let bin = dir.join("rec.bin");
        write_raw(&src, &bin, SampleFormat::F32, MemoryOrder::ChannelMajor, 1.0, 1000, |_, _| {}).unwrap();

        let list = sources(&bin).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!((list[0].id.as_str(), list[0].channels), (crate::core::sources::MAIN, 3));
        // Raw sidecars give gains in µV
        assert_eq!(list[0].unit, dsp_core::SignalUnit::Microvolt);
        assert_eq!(open_source(&bin, crate::core::sources::MAIN).unwrap().info().channel_count(), 3);
        assert!(open_source(&bin, "nope").is_err());
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
            crate::container::binary::codec::decode_run(info.format, &bytes[i * n * b..(i + 1) * n * b], &mut decoded[i * n..(i + 1) * n], ch.gain, ch.offset);
        }
        assert_eq!(decoded, values, "{}", info.name);
    }

    /// `read_native`, put back in channel-major order, equals `read` of every channel.
    pub(crate) fn assert_native_matches(rec: &dyn RecordingSource, samples: std::ops::Range<u64>) {
        let (nch, n) = (rec.info().channel_count(), (samples.end - samples.start) as usize);
        let all: Vec<usize> = (0..nch).collect();
        let mut values = vec![0.0f32; nch * n];
        rec.read(&all, samples.clone(), &mut values).unwrap();
        let mut native = vec![0.0f32; nch * n];
        let order = rec.read_native(samples, &mut native).unwrap();
        if order == MemoryOrder::TimeMajor {
            native = (0..nch).flat_map(|c| (0..n).map(move |t| (c, t))).map(|(c, t)| native[t * nch + c]).collect();
        }
        assert_eq!(native, values, "{} {order:?}", rec.info().name);
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
            assert_native_matches(rec.as_ref(), 100..1_900);
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
            assert_native_matches(open(&z).unwrap().as_ref(), 100..1_900);
            let cached = CachedRecording::wrap(open(&z).unwrap(), 1 << 20);
            assert_native_matches(cached.as_ref(), 100..1_900);
            assert_stored_matches(cached.as_ref(), &channels, 100..1_900);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
