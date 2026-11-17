//! A recording that grows: a source's blocks appended to a Zarr store while it is acquired.
//!
//! Layout (readable by [`crate::ZarrRecording`] at any moment of the run, and after a crash up to
//! the last chunk stored):
//! - root attributes `sample_rate_hz`, `channels`, `samples` (updated with every chunk),
//!   `channel_names`, `live` (true while recording);
//! - `/traces` `[channels, samples]` in the source's stored type (int16, int32, float32), with
//!   `gain` and `unit` attributes, chunked `[channels, chunk_samples]`; its length grows chunk by
//!   chunk;
//! - `/blocks/{first_sample, source_time_ns, received_unix_ns, lost_before}` (u64): one row per
//!   block as the source delivered it: the source's clock, the arrival time and the gaps.

use std::path::{Path, PathBuf};

use dspk_core::{DspError, DspResult, SampleFormat};
use serde_json::json;
use zarrs::array::{data_type, Array, ArrayBuilder};
use zarrs::group::{Group, GroupBuilder};
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

use super::{LiveBlock, LiveLayout};
use crate::container::zarr::open_store;
use crate::generic::zarr_traces::{compression_codecs, CHANNEL_NAMES_ATTR};
use crate::ZarrOptions;

/// The raw data's folder name in a recording's folder.
pub const RAW_ZARR: &str = "raw.zarr";

/// Rows per chunk of the `/blocks/*` arrays.
const BLOCK_ROWS_CHUNK: u64 = 4096;
/// The `/blocks/*` arrays, in the order of a row.
const BLOCK_COLUMNS: [&str; 4] = ["first_sample", "source_time_ns", "received_unix_ns", "lost_before"];

type Store = dyn ReadableWritableListableStorageTraits;

fn zerr(path: &Path, what: &str) -> impl Fn(String) -> DspError + use<> {
    let at = format!("{}: zarr {what}", path.display());
    move |e| DspError::Io(format!("{at}: {e}"))
}

pub struct LiveZarrWriter {
    path: PathBuf,
    store: ReadableWritableListableStorage,
    traces: Array<Store>,
    blocks: Vec<Array<Store>>,
    layout: LiveLayout,
    chunk: u64,
    /// Samples stored (whole chunks, until `finish`).
    stored: u64,
    /// The chunk being filled: channel-major stored bytes, row length `chunk`.
    pending: Vec<u8>,
    filled: u64,
    /// Block rows not stored yet, and rows stored.
    rows: Vec<[u64; 4]>,
    rows_stored: u64,
}

impl LiveZarrWriter {
    /// Creates `path` for `layout`'s data, chunked and compressed as `options` says.
    pub fn create(path: &Path, layout: &LiveLayout, options: &ZarrOptions) -> DspResult<Self> {
        let channels = layout.channel_count() as u64;
        if !matches!(layout.format, SampleFormat::I16 | SampleFormat::I32 | SampleFormat::F32) {
            return Err(DspError::UnsupportedFormat(format!("live recordings store int16, int32 or float32, not {}", layout.format.name())));
        }
        if path.exists() {
            return Err(DspError::Io(format!("{} exists already: a recording is never overwritten", path.display())));
        }
        std::fs::create_dir_all(path)?;
        let store = open_store(path)?;

        let mut root = GroupBuilder::new().build(store.clone(), "/").map_err(|e| zerr(path, "root group")(e.to_string()))?;
        let attrs = root.attributes_mut();
        attrs.insert("sample_rate_hz".into(), json!(layout.sample_rate.rate_hz()));
        attrs.insert("channels".into(), json!(channels));
        attrs.insert("samples".into(), json!(0));
        attrs.insert("live".into(), json!(true));
        attrs.insert("source".into(), json!(layout.name));
        attrs.insert(CHANNEL_NAMES_ATTR.into(), json!(layout.channels.iter().map(|c| c.name.clone()).collect::<Vec<_>>()));
        attrs.insert("created_by".into(), json!("dspk (dspk-io live, zarrs)"));
        root.store_metadata().map_err(|e| zerr(path, "root metadata")(e.to_string()))?;
        GroupBuilder::new().build(store.clone(), "/blocks").and_then(|g| g.store_metadata().map_err(Into::into)).map_err(|e| zerr(path, "/blocks")(e.to_string()))?;

        let chunk = options.chunk_samples.max(1) as u64;
        let mut builder = match layout.format {
            SampleFormat::I16 => ArrayBuilder::new(vec![channels, 0], vec![channels, chunk], data_type::int16(), 0i16),
            SampleFormat::I32 => ArrayBuilder::new(vec![channels, 0], vec![channels, chunk], data_type::int32(), 0i32),
            _ => ArrayBuilder::new(vec![channels, 0], vec![channels, chunk], data_type::float32(), 0.0f32),
        };
        builder.dimension_names(["channels", "samples"].into());
        builder.bytes_to_bytes_codecs(compression_codecs(options.compression)?);
        let mut traces = builder.build(store.clone(), "/traces").map_err(|e| zerr(path, "/traces")(e.to_string()))?;
        let first = layout.channels.first();
        traces.attributes_mut().insert("gain".into(), json!(first.map_or(1.0, |c| c.gain)));
        traces.attributes_mut().insert("unit".into(), serde_json::to_value(first.map(|c| c.unit.clone()).unwrap_or_default()).unwrap_or_default());
        traces.store_metadata().map_err(|e| zerr(path, "/traces metadata")(e.to_string()))?;

        let blocks = BLOCK_COLUMNS
            .iter()
            .map(|name| {
                let array = ArrayBuilder::new(vec![0], vec![BLOCK_ROWS_CHUNK], data_type::uint64(), 0u64)
                    .build(store.clone(), &format!("/blocks/{name}"))
                    .map_err(|e| zerr(path, name)(e.to_string()))?;
                array.store_metadata().map_err(|e| zerr(path, name)(e.to_string()))?;
                Ok(array)
            })
            .collect::<DspResult<Vec<_>>>()?;

        Ok(Self {
            path: path.to_path_buf(),
            store,
            traces,
            blocks,
            layout: layout.clone(),
            chunk,
            stored: 0,
            pending: Vec::new(),
            filled: 0,
            rows: Vec::new(),
            rows_stored: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Samples stored so far (readable from the disk).
    pub fn stored_samples(&self) -> u64 {
        self.stored
    }

    /// Samples per chunk.
    pub fn chunk_samples(&self) -> u64 {
        self.chunk
    }

    /// Appends a block (the next one: blocks arrive in order, `lost_before` counted in its
    /// `first_sample`). Lost samples are stored as zeros, so sample indices stay those of the source;
    /// `/blocks/lost_before` says where they are.
    pub fn push(&mut self, block: &LiveBlock) -> DspResult<()> {
        block.check(&self.layout)?;
        let head = self.stored + self.filled;
        if block.first_sample < head {
            return Err(DspError::InvalidConfig(format!("block at sample {} after sample {head}: blocks arrive in order", block.first_sample)));
        }
        let gap = block.first_sample - head;
        self.rows.push([block.first_sample, block.source_time_ns, block.received_unix_ns, block.lost_before.max(gap)]);
        if gap > 0 {
            let zeros = vec![0u8; self.layout.frame_bytes() * gap as usize];
            self.append(&zeros, gap as usize)?;
        }
        self.append(&block.data, block.samples)
    }

    /// Appends `samples` columns of channel-major stored bytes.
    fn append(&mut self, data: &[u8], samples: usize) -> DspResult<()> {
        let (channels, bytes) = (self.layout.channel_count(), self.layout.format.bytes());
        let chunk_row = self.chunk as usize * bytes;
        let mut col = 0usize;
        while col < samples {
            if self.pending.is_empty() {
                self.pending = vec![0u8; channels * chunk_row];
            }
            let take = (self.chunk - self.filled).min((samples - col) as u64) as usize;
            for c in 0..channels {
                let src = c * samples * bytes + col * bytes;
                let dst = c * chunk_row + self.filled as usize * bytes;
                self.pending[dst..dst + take * bytes].copy_from_slice(&data[src..src + take * bytes]);
            }
            self.filled += take as u64;
            col += take;
            if self.filled == self.chunk {
                self.store_pending(self.chunk)?;
            }
        }
        Ok(())
    }

    /// Stores the first `len` columns of the pending chunk, grows the array to cover them, and
    /// stores the block rows waiting.
    fn store_pending(&mut self, len: u64) -> DspResult<()> {
        if len == 0 {
            return Ok(());
        }
        let (channels, bytes) = (self.layout.channel_count() as u64, self.layout.format.bytes());
        let chunk_row = self.chunk as usize * bytes;
        // The chunk's first `len` columns, channel-major with row length `len`
        let mut values = Vec::with_capacity(channels as usize * len as usize * bytes);
        for c in 0..channels as usize {
            values.extend_from_slice(&self.pending[c * chunk_row..c * chunk_row + len as usize * bytes]);
        }
        let end = self.stored + len;
        self.traces.set_shape(vec![channels, end]).map_err(|e| zerr(&self.path, "grow /traces")(e.to_string()))?;
        let subset = [0..channels, self.stored..end];
        let write = zerr(&self.path, "write /traces");
        match self.layout.format {
            SampleFormat::I16 => self.traces.store_array_subset(&subset, decode::<2, i16>(&values, i16::from_le_bytes)),
            SampleFormat::I32 => self.traces.store_array_subset(&subset, decode::<4, i32>(&values, i32::from_le_bytes)),
            _ => self.traces.store_array_subset(&subset, decode::<4, f32>(&values, f32::from_le_bytes)),
        }
        .map_err(|e| write(e.to_string()))?;
        self.traces.store_metadata().map_err(|e| zerr(&self.path, "/traces metadata")(e.to_string()))?;
        self.stored = end;
        self.pending.clear();
        self.filled = 0;
        self.store_rows()?;
        self.update_root()
    }

    fn store_rows(&mut self) -> DspResult<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let (start, end) = (self.rows_stored, self.rows_stored + self.rows.len() as u64);
        for (k, array) in self.blocks.iter_mut().enumerate() {
            let column: Vec<u64> = self.rows.iter().map(|r| r[k]).collect();
            array.set_shape(vec![end]).map_err(|e| zerr(&self.path, "grow /blocks")(e.to_string()))?;
            array.store_array_subset(&[start..end], column).map_err(|e| zerr(&self.path, "write /blocks")(e.to_string()))?;
            array.store_metadata().map_err(|e| zerr(&self.path, "/blocks metadata")(e.to_string()))?;
        }
        self.rows_stored = end;
        self.rows.clear();
        Ok(())
    }

    fn update_root(&self) -> DspResult<()> {
        let mut root = Group::open(self.store.clone(), "/").map_err(|e| zerr(&self.path, "root group")(e.to_string()))?;
        root.attributes_mut().insert("samples".into(), json!(self.stored));
        root.store_metadata().map_err(|e| zerr(&self.path, "root metadata")(e.to_string()))
    }

    /// Stores what is pending (the last, shorter chunk) and marks the recording finished.
    pub fn finish(mut self) -> DspResult<u64> {
        self.store_pending(self.filled)?;
        self.store_rows()?;
        let mut root = Group::open(self.store.clone(), "/").map_err(|e| zerr(&self.path, "root group")(e.to_string()))?;
        root.attributes_mut().insert("samples".into(), json!(self.stored));
        root.attributes_mut().insert("live".into(), json!(false));
        root.store_metadata().map_err(|e| zerr(&self.path, "root metadata")(e.to_string()))?;
        Ok(self.stored)
    }
}

/// Little-endian bytes → values.
fn decode<const B: usize, T>(bytes: &[u8], from: fn([u8; B]) -> T) -> Vec<T> {
    bytes.as_chunks::<B>().0.iter().map(|b| from(*b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::{Generator, GeneratorParams, LiveSource};
    use crate::ZarrRecording;
    use dspk_core::RecordingSource;

    #[test]
    fn grows_chunk_by_chunk_and_reads_back_with_gaps_as_zeros() {
        let dir = std::env::temp_dir().join(format!("dspk_live_zarr_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let params = GeneratorParams { channels: 3, sample_rate_hz: 1000.0, block_ms: 7.0, ..Default::default() };
        let generator = Generator::new("gen", &params).unwrap();
        let layout = generator.layout().clone();
        let path = dir.join(RAW_ZARR);
        let mut writer = LiveZarrWriter::create(&path, &layout, &ZarrOptions { chunk_samples: 20, ..Default::default() }).unwrap();
        let mut blocks = Vec::new();
        for first in [0u64, 7, 14, 21, 35, 42] {
            // 28..35 lost
            let mut b = generator.block_at(first, 7).unwrap();
            if first == 35 {
                b.lost_before = 7;
            }
            writer.push(&b).unwrap();
            blocks.push(b);
        }
        assert_eq!(writer.stored_samples(), 40, "two whole chunks stored while recording");
        assert_eq!(ZarrRecording::open(&path).unwrap().info().samples, 40, "readable during the run");
        assert_eq!(writer.finish().unwrap(), 49);

        let rec = ZarrRecording::open(&path).unwrap();
        assert_eq!(rec.info().samples, 49);
        assert_eq!(rec.info().format, SampleFormat::I16);
        let mut out = vec![0.0f32; 3 * 49];
        rec.read(&[0, 1, 2], 0..49, &mut out).unwrap();
        for b in &blocks {
            let values = b.to_f32(&layout);
            for c in 0..3 {
                for s in 0..7 {
                    assert_eq!(out[c * 49 + b.first_sample as usize + s], values[c * 7 + s]);
                }
            }
        }
        assert!(out[28..35].iter().all(|&v| v == 0.0), "the gap is zeros");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
