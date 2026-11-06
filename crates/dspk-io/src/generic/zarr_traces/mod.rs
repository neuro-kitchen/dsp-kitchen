//! Zarr v3 recordings (`zarrs`), read by chunk.
//!
//! Layout written by [`write_zarr`]: root group attributes `sample_rate_hz`, `channels`,
//! `samples`; array `/traces` of shape `[channels, samples]` (`float32`, or `int16` with a
//! `gain` array attribute), chunked `[channels, chunk_samples]`, with a `unit` array attribute
//! (`"microvolt"`, …; a legacy `gain_uv` attribute means µV). Arrays stored
//! `[samples, channels]` (dimension names `["samples", "channels"]` or `["time", ...]`) are
//! read too.

use std::ops::Range;
use std::path::Path;

use dspk_core::recording::{check_read, check_read_stored};
use dspk_core::{DspError, DspResult, MemoryOrder, RecordingInfo, RecordingSource, SampleFormat, SampleRate, SignalUnit};
use serde_json::{json, Value};
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, data_type};

use crate::container::binary::codec::{native_to_le, scale_frames, select_stored};
use crate::container::zarr::open_store;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::group::{Group, GroupBuilder};

mod format;
pub use format::ZarrTraces;

const ARRAY_PATH: &str = "/traces";
/// Default chunk length: ~0.17 s at 30 kHz.
pub const DEFAULT_CHUNK_SAMPLES: usize = 5000;

fn err(what: &str) -> impl Fn(String) -> DspError + '_ {
    move |e| DspError::Io(format!("zarr {what}: {e}"))
}

/// Chunked Zarr v3 recording.
pub struct ZarrRecording {
    info: RecordingInfo,
    array: Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
    /// Samples per chunk along time (the first chunk's).
    chunk_samples: Option<u64>,
}

impl ZarrRecording {
    pub fn open(path: &Path) -> DspResult<Self> {
        let store = open_store(path)?;
        let array = Array::open(store.clone(), ARRAY_PATH).map_err(|e| err("open /traces")(e.to_string()))?;
        let root = Group::open(store, "/").map_err(|e| err("open root group")(e.to_string()))?;

        // Data type and dimension names straight from the array metadata document
        let meta: Value = std::fs::read_to_string(path.join("traces").join("zarr.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Value::Null);
        let format = match meta["data_type"].as_str() {
            None => SampleFormat::F32,
            Some(name) => SampleFormat::parse(name).ok_or_else(|| DspError::UnsupportedFormat(format!("zarr data type {name}")))?,
        };
        let first_dim = meta["dimension_names"][0].as_str().unwrap_or("channels");
        let order = if matches!(first_dim, "samples" | "time" | "frames") { MemoryOrder::TimeMajor } else { MemoryOrder::ChannelMajor };

        let shape = array.shape();
        if shape.len() != 2 {
            return Err(DspError::UnsupportedFormat(format!("expected a 2D traces array, got {}D", shape.len())));
        }
        let (channels, samples) = match order {
            MemoryOrder::ChannelMajor => (shape[0] as usize, shape[1]),
            MemoryOrder::TimeMajor => (shape[1] as usize, shape[0]),
        };

        let attrs = root.attributes();
        let rate = attrs
            .get("sample_rate_hz")
            .or_else(|| attrs.get("sampling_frequency"))
            .and_then(Value::as_f64)
            .ok_or_else(|| DspError::InvalidConfig(format!("{} has no sample_rate_hz attribute", path.display())))?;
        // `gain` + `unit` attributes; a legacy `gain_uv` declares µV; neither: values as stored
        let array_attrs = array.attributes();
        let unit_attr = array_attrs.get("unit").and_then(|u| serde_json::from_value::<SignalUnit>(u.clone()).ok());
        let (gain, unit) = match (array_attrs.get("gain").and_then(Value::as_f64), array_attrs.get("gain_uv").and_then(Value::as_f64)) {
            (Some(g), _) => (g as f32, unit_attr.unwrap_or_default()),
            (None, Some(g)) => (g as f32, SignalUnit::Microvolt),
            (None, None) => (1.0, unit_attr.unwrap_or_default()),
        };

        let name = path.file_name().map_or_else(|| "recording.zarr".into(), |n| n.to_string_lossy().into_owned());
        let mut info = RecordingInfo::new(name, channels, samples, SampleRate::new(rate)?, format, order).with_gain(gain, unit);
        if let Some(names) = attrs.get(CHANNEL_NAMES_ATTR).and_then(Value::as_array).filter(|n| n.len() == channels) {
            for (ch, n) in info.channels.iter_mut().zip(names) {
                if let Some(n) = n.as_str() {
                    ch.name = n.to_string();
                }
            }
        }
        info.metadata.insert("format".into(), "zarr v3".into());
        let time_axis = if order == MemoryOrder::TimeMajor { 0 } else { 1 };
        let chunk_samples = array.chunk_shape(&[0, 0]).ok().map(|c| c[time_axis].get());
        Ok(Self { info, array, chunk_samples })
    }

    /// Stored values of channels `ch` × `samples`, channel-major.
    fn retrieve(&self, ch: Range<u64>, samples: Range<u64>) -> DspResult<Vec<f32>> {
        let values = self.retrieve_native(ch.clone(), samples.clone())?;
        if self.info.order == MemoryOrder::ChannelMajor {
            return Ok(values);
        }
        // [samples, channels] → [channels, samples]
        let (nc, ns) = ((ch.end - ch.start) as usize, (samples.end - samples.start) as usize);
        let mut t = vec![0.0f32; values.len()];
        for s in 0..ns {
            for c in 0..nc {
                t[c * ns + s] = values[s * nc + c];
            }
        }
        Ok(t)
    }

    /// Stored values of channels `ch` × `samples` in the array's order.
    fn retrieve_native(&self, ch: Range<u64>, samples: Range<u64>) -> DspResult<Vec<f32>> {
        let subset = match self.info.order {
            MemoryOrder::ChannelMajor => [ch.clone(), samples.clone()],
            MemoryOrder::TimeMajor => [samples.clone(), ch.clone()],
        };
        let read = |e: zarrs::array::ArrayError| err("read")(e.to_string());
        macro_rules! get {
            ($t:ty) => {
                self.array.retrieve_array_subset::<Vec<$t>>(&subset).map_err(read)?.into_iter().map(|v| v as f32).collect()
            };
        }
        let values: Vec<f32> = match self.info.format {
            SampleFormat::F32 => self.array.retrieve_array_subset::<Vec<f32>>(&subset).map_err(read)?,
            SampleFormat::I8 => get!(i8),
            SampleFormat::I16 => get!(i16),
            SampleFormat::U16 => get!(u16),
            SampleFormat::I32 => get!(i32),
            SampleFormat::F64 => get!(f64),
        };
        Ok(values)
    }
}

impl RecordingSource for ZarrRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn chunk_samples(&self) -> Option<u64> {
        self.chunk_samples
    }

    fn read_native(&self, samples: Range<u64>, out: &mut [f32]) -> DspResult<MemoryOrder> {
        let nch = self.info.channels.len();
        let all: Vec<usize> = (0..nch).collect();
        let n = check_read(&self.info, &all, &samples, out.len())?;
        if n == 0 || nch == 0 {
            return Ok(self.info.order);
        }
        let block = self.retrieve_native(0..nch as u64, samples)?;
        out.copy_from_slice(&block);
        match self.info.order {
            MemoryOrder::TimeMajor => scale_frames(&self.info, out),
            MemoryOrder::ChannelMajor => {
                for (row, c) in out.chunks_exact_mut(n).zip(&self.info.channels) {
                    row.iter_mut().for_each(|v| *v = *v * c.gain + c.offset);
                }
            }
        }
        Ok(self.info.order)
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        let n = check_read_stored(&self.info, channels, &samples, out.len())?;
        if n == 0 || channels.is_empty() {
            return Ok(());
        }
        let lo = *channels.iter().min().unwrap();
        let hi = *channels.iter().max().unwrap() + 1;
        let time_major = self.info.order == MemoryOrder::TimeMajor;
        let subset = if time_major { [samples.clone(), lo as u64..hi as u64] } else { [lo as u64..hi as u64, samples.clone()] };
        let bytes = self.info.format.bytes();
        let mut block = self
            .array
            .retrieve_array_subset::<ArrayBytes<'static>>(&subset)
            .map_err(|e| err("read")(e.to_string()))?
            .into_fixed()
            .map_err(|e| err("read")(e.to_string()))?
            .into_owned();
        native_to_le(&mut block, bytes);
        let cols: Vec<usize> = channels.iter().map(|&c| c - lo).collect();
        select_stored(&block, time_major, hi - lo, n, &cols, bytes, out);
        Ok(())
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        if n == 0 || channels.is_empty() {
            return Ok(());
        }
        // One subset spanning the requested channels, then pick rows
        let lo = *channels.iter().min().unwrap();
        let hi = *channels.iter().max().unwrap() + 1;
        let block = self.retrieve(lo as u64..hi as u64, samples)?;
        for (dst, &ch) in out.chunks_exact_mut(n).zip(channels) {
            let c = &self.info.channels[ch];
            let row = &block[(ch - lo) * n..(ch - lo + 1) * n];
            for (o, &v) in dst.iter_mut().zip(row) {
                *o = v * c.gain + c.offset;
            }
        }
        Ok(())
    }
}

/// How Zarr chunks are compressed (a bytes-to-bytes codec of the array).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZarrCompression {
    /// Uncompressed.
    None,
    /// The Zarr v3 `zstd` codec at `level` (zstd's levels; higher is smaller and slower).
    Zstd { level: i32 },
    /// The Zarr v3 `blosc` codec: LZ4 at `level` (0–9) after a byte shuffle of the float32
    /// samples (faster than zstd, usually a little larger).
    BloscLz4 { level: u8 },
}

/// The compression of [`ZarrOptions::default`]: zstd level 3, the codec Zarr's own tools default
/// to, fast enough to keep up with the device when chunks are compressed in parallel.
pub const DEFAULT_ZARR_COMPRESSION: ZarrCompression = ZarrCompression::Zstd { level: 3 };

/// Root attribute holding the channels' names (a list, one per channel).
pub const CHANNEL_NAMES_ATTR: &str = "channel_names";

/// Layout of a written Zarr recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZarrOptions {
    /// Samples per chunk along time (chunks span every channel). Several chunks per block of
    /// [`ZarrWriter::push`] are compressed and stored in parallel.
    pub chunk_samples: usize,
    pub compression: ZarrCompression,
    /// Compress and store the completed chunks of each block on several threads (default
    /// [`DEFAULT_PARALLEL_WRITES`]); `false`: one after the other on the calling thread.
    pub parallel: bool,
}

/// Default of [`ZarrOptions::parallel`]: chunks are compressed and stored in parallel.
pub const DEFAULT_PARALLEL_WRITES: bool = true;

impl Default for ZarrOptions {
    fn default() -> Self {
        Self { chunk_samples: DEFAULT_CHUNK_SAMPLES, compression: DEFAULT_ZARR_COMPRESSION, parallel: DEFAULT_PARALLEL_WRITES }
    }
}

/// Streams `source` into a new Zarr v3 store (`float32`, in the source's unit: the first
/// channel's), laid out as `options` says.
pub fn write_zarr(source: &dyn RecordingSource, path: &Path, options: &ZarrOptions, mut progress: impl FnMut(u64, u64)) -> DspResult<()> {
    let info = source.info();
    let (nch, total) = (info.channels.len(), info.samples);
    let unit = info.channels.first().map(|c| c.unit.clone()).unwrap_or_default();
    let mut writer = ZarrWriter::create(path, nch, total, info.sample_rate_hz(), &unit, options)?;
    writer.set_channel_names(&info.channels.iter().map(|c| c.name.clone()).collect::<Vec<_>>())?;
    // Read several chunks at a time, so the writer stores them in parallel
    let threads = if options.parallel { rayon::current_num_threads().max(1) } else { 1 };
    let block = writer.chunk_samples() * threads as u64;
    let channels: Vec<usize> = (0..nch).collect();
    let mut buf = Vec::new();
    let mut s0 = 0;
    while s0 < total {
        let s1 = (s0 + block).min(total);
        let len = (s1 - s0) as usize;
        buf.resize(nch * len, 0.0);
        source.read(&channels, s0..s1, &mut buf)?;
        writer.push(&buf, len, 0..len)?;
        progress(s1, total);
        s0 = s1;
    }
    writer.finish()
}

/// Writes a new Zarr recording in the layout [`write_zarr`] gives (`float32`
/// `[channels, samples]`, chunked `[channels, chunk_samples]`, compressed per [`ZarrOptions`])
/// from blocks of columns arriving in order, e.g. the windows of a pipeline run: [`Self::push`]
/// fills chunks and, at the end of each call, compresses and stores the chunks it completed in
/// parallel (rayon; every chunk is a separate file, so the stores are independent). The data never
/// goes through a temporary file.
pub struct ZarrWriter {
    store: ReadableWritableListableStorage,
    array: Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
    channels: usize,
    total: u64,
    chunk: u64,
    /// Samples handed to the store so far (a multiple of `chunk`, until the end).
    stored: u64,
    parallel: bool,
    /// The chunk being filled, channel-major with row length `chunk_len()`.
    pending: Vec<f32>,
    filled: usize,
    /// Completed chunks not stored yet: first sample, values.
    ready: Vec<(u64, Vec<f32>)>,
}

impl ZarrWriter {
    /// Creates the store at `path` for `channels × samples` values at `sample_rate_hz` in `unit`.
    pub fn create(path: &Path, channels: usize, samples: u64, sample_rate_hz: f64, unit: &SignalUnit, options: &ZarrOptions) -> DspResult<Self> {
        std::fs::create_dir_all(path)?;
        let store = open_store(path)?;

        let mut root = GroupBuilder::new().build(store.clone(), "/").map_err(|e| err("root group")(e.to_string()))?;
        let attrs = root.attributes_mut();
        attrs.insert("sample_rate_hz".into(), json!(sample_rate_hz));
        attrs.insert("channels".into(), json!(channels));
        attrs.insert("samples".into(), json!(samples));
        attrs.insert("created_by".into(), json!("dspk (dspk-io, zarrs)"));
        root.store_metadata().map_err(|e| err("root metadata")(e.to_string()))?;

        let chunk = (options.chunk_samples.max(1) as u64).min(samples.max(1));
        let mut builder = ArrayBuilder::new(vec![channels as u64, samples], vec![channels as u64, chunk], data_type::float32(), 0.0f32);
        builder.dimension_names(["channels", "samples"].into());
        builder.bytes_to_bytes_codecs(compression_codecs(options.compression)?);
        let mut array = builder.build(store.clone(), ARRAY_PATH).map_err(|e| err("array")(e.to_string()))?;
        array.attributes_mut().insert("unit".into(), serde_json::to_value(unit).map_err(|e| err("unit")(e.to_string()))?);
        array.store_metadata().map_err(|e| err("array metadata")(e.to_string()))?;

        Ok(Self { store, array, channels, total: samples, chunk, stored: 0, parallel: options.parallel, pending: Vec::new(), filled: 0, ready: Vec::new() })
    }

    /// Records the channels' names (the `channel_names` attribute of the root group; read back
    /// as each channel's name).
    pub fn set_channel_names(&mut self, names: &[String]) -> DspResult<()> {
        if names.len() != self.channels {
            return Err(DspError::ShapeMismatch { expected: vec![self.channels], actual: vec![names.len()] });
        }
        let mut root = Group::open(self.store.clone(), "/").map_err(|e| err("root group")(e.to_string()))?;
        root.attributes_mut().insert(CHANNEL_NAMES_ATTR.into(), json!(names));
        root.store_metadata().map_err(|e| err("root metadata")(e.to_string()))
    }

    /// Samples per chunk.
    pub fn chunk_samples(&self) -> u64 {
        self.chunk
    }

    /// First sample of the chunk being filled.
    fn chunk_start(&self) -> u64 {
        self.stored + self.ready.len() as u64 * self.chunk
    }

    /// Length of the chunk being filled (the last one may be shorter).
    fn chunk_len(&self) -> usize {
        self.chunk.min(self.total - self.chunk_start()) as usize
    }

    /// Appends columns `cols` of the channel-major block `rows` (`[channels, row_len]`), then
    /// stores the chunks this completed, in parallel.
    ///
    /// # Errors
    /// [`DspError::ShapeMismatch`] for a block of the wrong size or past the end of the array;
    /// [`DspError::Io`] when a chunk cannot be stored.
    pub fn push(&mut self, rows: &[f32], row_len: usize, cols: Range<usize>) -> DspResult<()> {
        if rows.len() != self.channels * row_len || cols.end > row_len {
            return Err(DspError::ShapeMismatch { expected: vec![self.channels, row_len], actual: vec![rows.len(), cols.end] });
        }
        let mut col = cols.start;
        while col < cols.end {
            let len = self.chunk_len();
            if len == 0 {
                return Err(DspError::ShapeMismatch {
                    expected: vec![self.channels, self.total as usize],
                    actual: vec![self.channels, self.total as usize + (cols.end - col)],
                });
            }
            if self.pending.is_empty() {
                self.pending = vec![0.0; self.channels * len];
            }
            let take = (len - self.filled).min(cols.end - col);
            for c in 0..self.channels {
                let src = &rows[c * row_len + col..c * row_len + col + take];
                self.pending[c * len + self.filled..c * len + self.filled + take].copy_from_slice(src);
            }
            self.filled += take;
            col += take;
            if self.filled == len {
                let start = self.chunk_start();
                self.ready.push((start, std::mem::take(&mut self.pending)));
                self.filled = 0;
            }
        }
        self.store_ready()
    }

    /// Compresses and stores the completed chunks, in parallel unless [`ZarrOptions::parallel`] is
    /// off.
    fn store_ready(&mut self) -> DspResult<()> {
        use rayon::prelude::*;
        let (array, channels, total) = (&self.array, self.channels as u64, self.total);
        let store = |(start, values): &(u64, Vec<f32>)| {
            let end = (start + values.len() as u64 / channels).min(total);
            array.store_array_subset(&[0..channels, *start..end], values.as_slice()).map_err(|e| err("write")(e.to_string()))
        };
        if self.parallel {
            self.ready.par_iter().try_for_each(store)?;
        } else {
            self.ready.iter().try_for_each(store)?;
        }
        self.stored += self.ready.iter().map(|(_, v)| v.len() as u64 / channels).sum::<u64>();
        self.ready.clear();
        Ok(())
    }

    /// Checks that every sample was written.
    ///
    /// # Errors
    /// [`DspError::ShapeMismatch`] when fewer than `samples` columns were pushed.
    pub fn finish(self) -> DspResult<()> {
        if self.stored != self.total {
            return Err(DspError::ShapeMismatch {
                expected: vec![self.channels, self.total as usize],
                actual: vec![self.channels, (self.stored as usize) + self.filled],
            });
        }
        Ok(())
    }
}

/// The bytes-to-bytes codecs of `compression`.
fn compression_codecs(compression: ZarrCompression) -> DspResult<Vec<std::sync::Arc<dyn zarrs::array::BytesToBytesCodecTraits>>> {
    use zarrs::array::codec::{BloscCodec, BloscCompressionLevel, BloscCompressor, BloscShuffleMode, ZstdCodec};
    Ok(match compression {
        ZarrCompression::None => Vec::new(),
        ZarrCompression::Zstd { level } => vec![std::sync::Arc::new(ZstdCodec::new(level, false))],
        ZarrCompression::BloscLz4 { level } => {
            let level = BloscCompressionLevel::try_from(level).map_err(|_| DspError::InvalidConfig(format!("blosc level {level} is not in 0..=9")))?;
            let codec = BloscCodec::new(BloscCompressor::LZ4, level, None, BloscShuffleMode::Shuffle, Some(size_of::<f32>()))
                .map_err(|e| err("blosc codec")(e.to_string()))?;
            vec![std::sync::Arc::new(codec)]
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_core::MemoryRecording;

    #[test]
    fn test_zarr_roundtrip_chunked() {
        let path = std::env::temp_dir().join(format!("dspk_io_zarr_{}.zarr", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);

        let data: Vec<f32> = (0..4).flat_map(|c| (0..1200).map(move |s| (c * 10_000 + s) as f32)).collect();
        let src = MemoryRecording::new("src", data, 4, 20_000.0).unwrap();
        write_zarr(&src, &path, &ZarrOptions { chunk_samples: 500, ..Default::default() }, |_, _| {}).unwrap();

        let rec = ZarrRecording::open(&path).unwrap();
        assert_eq!(rec.info().channel_count(), 4);
        assert_eq!(rec.info().samples, 1200);
        assert_eq!(rec.info().sample_rate_hz(), 20_000.0);

        // Crosses a chunk boundary, channels out of order
        let mut out = vec![0.0; 2 * 4];
        rec.read(&[3, 1], 498..502, &mut out).unwrap();
        assert_eq!(out, vec![30498.0, 30499.0, 30500.0, 30501.0, 10498.0, 10499.0, 10500.0, 10501.0]);
        let _ = std::fs::remove_dir_all(&path);
    }

    /// Blocks that start and end anywhere relative to the chunks (and only a column range of each
    /// block) give the same array; pushing past the end or finishing short is an error.
    #[test]
    fn test_zarr_writer_blocks_across_chunks() {
        let path = std::env::temp_dir().join(format!("dspk_io_zarr_writer_{}.zarr", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let (nch, total) = (3usize, 1_234u64);
        let value = |c: usize, s: u64| (c * 10_000) as f32 + s as f32;
        let options = ZarrOptions { chunk_samples: 500, ..Default::default() };
        let mut writer = ZarrWriter::create(&path, nch, total, 30_000.0, &SignalUnit::Microvolt, &options).unwrap();
        let mut s0 = 0u64;
        for len in [7u64, 493, 1, 499, 234] {
            // Block of `len + 4` columns with 2 halo columns each side; only the middle is pushed
            let row_len = len as usize + 4;
            let rows: Vec<f32> = (0..nch).flat_map(|c| (0..row_len).map(move |t| value(c, (s0 + t as u64).wrapping_sub(2)))).collect();
            writer.push(&rows, row_len, 2..2 + len as usize).unwrap();
            s0 += len;
        }
        assert_eq!(s0, total);
        assert!(writer.push(&vec![0.0; nch], 1, 0..1).is_err(), "past the end");
        writer.finish().unwrap();

        let rec = ZarrRecording::open(&path).unwrap();
        let mut out = vec![0.0; nch * total as usize];
        rec.read(&[0, 1, 2], 0..total, &mut out).unwrap();
        let expected: Vec<f32> = (0..nch).flat_map(|c| (0..total).map(move |s| value(c, s))).collect();
        assert_eq!(out, expected);

        let short = ZarrWriter::create(&path.with_extension("short.zarr"), nch, total, 30_000.0, &SignalUnit::Microvolt, &options).unwrap();
        assert!(short.finish().is_err(), "nothing written");
        let _ = std::fs::remove_dir_all(&path);
        let _ = std::fs::remove_dir_all(path.with_extension("short.zarr"));
    }
}
