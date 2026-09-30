//! Zarr v3 recordings (`zarrs`), read by chunk.
//!
//! Layout written by [`write_zarr`]: root group attributes `sample_rate_hz`, `channels`,
//! `samples`; array `/traces` of shape `[channels, samples]` (`float32` or `int16` with a
//! `gain_uv` array attribute), chunked `[channels, chunk_samples]`. Arrays stored
//! `[samples, channels]` (dimension names `["samples", "channels"]` or `["time", ...]`) are
//! read too.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use dsp_core::recording::{check_read, check_read_stored};
use dsp_core::{DspError, DspResult, MemoryOrder, RecordingInfo, RecordingSource, SampleFormat, SampleRate};
use serde_json::{json, Value};
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, data_type};

use crate::codec::{native_to_le, select_stored};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::{Group, GroupBuilder};
use zarrs::storage::ReadableWritableListableStorage;

const ARRAY_PATH: &str = "/traces";
/// Default chunk length: ~0.17 s at 30 kHz.
pub const DEFAULT_CHUNK_SAMPLES: usize = 5000;

fn err(what: &str) -> impl Fn(String) -> DspError + '_ {
    move |e| DspError::Io(format!("zarr {what}: {e}"))
}

fn open_store(path: &Path) -> DspResult<ReadableWritableListableStorage> {
    Ok(Arc::new(FilesystemStore::new(path).map_err(|e| err("store")(e.to_string()))?))
}

/// Chunked Zarr v3 recording.
pub struct ZarrRecording {
    info: RecordingInfo,
    array: Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>,
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
        let gain = array.attributes().get("gain_uv").and_then(Value::as_f64).unwrap_or(1.0) as f32;

        let name = path.file_name().map_or_else(|| "recording.zarr".into(), |n| n.to_string_lossy().into_owned());
        let mut info = RecordingInfo::new(name, channels, samples, SampleRate::new(rate)?, format, order).with_gain_uv(gain);
        info.metadata.insert("format".into(), "zarr v3".into());
        Ok(Self { info, array })
    }

    /// Stored values of channels `ch` × `samples`, channel-major.
    fn retrieve(&self, ch: Range<u64>, samples: Range<u64>) -> DspResult<Vec<f32>> {
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
}

impl RecordingSource for ZarrRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
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
                *o = v * c.gain_uv + c.offset_uv;
            }
        }
        Ok(())
    }
}

/// Streams `source` into a new Zarr v3 store (`float32`, µV), `chunk_samples` per chunk.
pub fn write_zarr(
    source: &dyn RecordingSource,
    path: &Path,
    chunk_samples: usize,
    mut progress: impl FnMut(u64, u64),
) -> DspResult<()> {
    let info = source.info();
    let (nch, total) = (info.channels.len(), info.samples);
    std::fs::create_dir_all(path)?;
    let store = open_store(path)?;

    let mut root = GroupBuilder::new().build(store.clone(), "/").map_err(|e| err("root group")(e.to_string()))?;
    let attrs = root.attributes_mut();
    attrs.insert("sample_rate_hz".into(), json!(info.sample_rate_hz()));
    attrs.insert("channels".into(), json!(nch));
    attrs.insert("samples".into(), json!(total));
    attrs.insert("created_by".into(), json!("dsp-kitchen (dsp-io, zarrs)"));
    root.store_metadata().map_err(|e| err("root metadata")(e.to_string()))?;

    let chunk = (chunk_samples.max(1) as u64).min(total.max(1));
    let array = ArrayBuilder::new(vec![nch as u64, total], vec![nch as u64, chunk], data_type::float32(), 0.0f32)
        .dimension_names(["channels", "samples"].into())
        .build(store, ARRAY_PATH)
        .map_err(|e| err("array")(e.to_string()))?;
    array.store_metadata().map_err(|e| err("array metadata")(e.to_string()))?;

    let channels: Vec<usize> = (0..nch).collect();
    let mut buf = Vec::new();
    let mut s0 = 0;
    while s0 < total {
        let s1 = (s0 + chunk).min(total);
        buf.resize(nch * (s1 - s0) as usize, 0.0);
        source.read(&channels, s0..s1, &mut buf)?;
        array.store_array_subset(&[0..nch as u64, s0..s1], buf.as_slice()).map_err(|e| err("write")(e.to_string()))?;
        progress(s1, total);
        s0 = s1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::MemoryRecording;

    #[test]
    fn test_zarr_roundtrip_chunked() {
        let path = std::env::temp_dir().join(format!("dsp_io_zarr_{}.zarr", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);

        let data: Vec<f32> = (0..4).flat_map(|c| (0..1200).map(move |s| (c * 10_000 + s) as f32)).collect();
        let src = MemoryRecording::new("src", data, 4, 20_000.0).unwrap();
        write_zarr(&src, &path, 500, |_, _| {}).unwrap();

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
}
