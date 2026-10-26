//! NWB files stored as Zarr v3 (hdmf-zarr layout), read one series at a time.
//!
//! An NWB store holds many signals; this opens one continuous series (`ElectricalSeries` or
//! `TimeSeries` with `starting_time` + `rate`) as a recording. By default the largest
//! `ElectricalSeries` in `/acquisition` is chosen; [`NwbZarrRecording::open_series`] picks
//! another. Data is `[time, channel]` (or `[time]`); values are scaled by the series'
//! `conversion` (volts → µV for electrical series).

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use dsp_core::recording::{check_read, check_read_stored};
use dsp_core::{DspError, DspResult, MemoryOrder, RationalTime, RecordingInfo, RecordingSource, SampleFormat, SampleRate, SignalUnit, MICROVOLTS_PER_VOLT};
use serde_json::Value;
use zarrs::array::{Array, ArrayBytes};

use crate::container::binary::codec::{native_to_le, scale_frames, select_stored};
use crate::container::zarr::{read_all, read_node_json};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableStorageTraits;

type Storage = dyn ReadableStorageTraits;

fn err(what: impl std::fmt::Display) -> DspError {
    DspError::Io(format!("nwb-zarr: {what}"))
}

/// True when `path` is a Zarr store whose root is an NWB file.
pub fn is_nwb_zarr(path: &Path) -> bool {
    read_node_json(path, "").is_some_and(|m| m["attributes"]["neurodata_type"] == "NWBFile")
}

/// A continuous series found in the store.
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesEntry {
    /// Path inside the store, e.g. `/acquisition/HDEMG`.
    pub path: String,
    pub neurodata_type: String,
    pub channels: usize,
    pub samples: u64,
    pub rate: f64,
    pub start_time: f64,
    /// Stored sample type.
    pub format: SampleFormat,
    /// Unit of the stored data (`volts`, `a.u.`, …).
    pub unit: String,
}

/// Continuous series under `/acquisition` that have a regular rate and a readable data type
/// (irregular, timestamped series such as event trains are not recordings).
pub fn list_series(store: &Path) -> Vec<SeriesEntry> {
    let Ok(entries) = std::fs::read_dir(store.join("acquisition")) else { return Vec::new() };
    let mut out: Vec<SeriesEntry> = entries
        .flatten()
        .filter_map(|e| {
            let path = format!("/acquisition/{}", e.file_name().to_string_lossy());
            let meta = read_node_json(store, &path)?;
            let kind = meta["attributes"]["neurodata_type"].as_str()?.to_string();
            if !matches!(kind.as_str(), "ElectricalSeries" | "TimeSeries") {
                return None;
            }
            let start_meta = read_node_json(store, &format!("{path}/starting_time"))?;
            let rate = start_meta["attributes"]["rate"].as_f64()?;
            let data_meta = read_node_json(store, &format!("{path}/data"))?;
            let unit = data_meta["attributes"]["unit"].as_str().unwrap_or("a.u.").to_string();
            let format = SampleFormat::parse(data_meta["data_type"].as_str()?)?;
            let shape: Vec<u64> = data_meta["shape"].as_array()?.iter().filter_map(Value::as_u64).collect();
            let (samples, channels) = match shape[..] {
                [t] => (t, 1),
                [t, c] => (t, c as usize),
                _ => return None,
            };
            let fs: Arc<Storage> = Arc::new(FilesystemStore::new(store).ok()?);
            let start_time = read_all::<f64>(&fs, &format!("{path}/starting_time")).and_then(|v| v.first().copied()).unwrap_or(0.0);
            Some(SeriesEntry { path, neurodata_type: kind, channels, samples, rate, start_time, format, unit })
        })
        .collect();
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

pub struct NwbZarrRecording {
    info: RecordingInfo,
    data: Array<Storage>,
    two_d: bool,
    /// Samples per chunk along time (the first chunk's).
    chunk_samples: Option<u64>,
}

impl NwbZarrRecording {
    /// Opens the largest `ElectricalSeries`, or the largest continuous series if there is none.
    pub fn open(path: &Path) -> DspResult<Self> {
        let series = list_series(path);
        let size = |s: &&SeriesEntry| s.samples * s.channels as u64;
        let pick = series
            .iter()
            .filter(|s| s.neurodata_type == "ElectricalSeries")
            .max_by_key(size)
            .or_else(|| series.iter().max_by_key(size))
            .ok_or_else(|| err(format!("{} has no continuous series in /acquisition", path.display())))?;
        Self::open_series(path, &pick.path)
    }

    /// Opens the series at `series` (e.g. `/acquisition/HDEMG`).
    pub fn open_series(path: &Path, series: &str) -> DspResult<Self> {
        let store: Arc<Storage> = Arc::new(FilesystemStore::new(path).map_err(err)?);
        let data_path = format!("{series}/data");
        let data = Array::open(store.clone(), &data_path).map_err(|e| err(format!("{data_path}: {e}")))?;
        let meta = read_node_json(path, &data_path).ok_or_else(|| err(format!("{data_path}: no metadata")))?;
        let group = read_node_json(path, series).unwrap_or(Value::Null);
        let kind = group["attributes"]["neurodata_type"].as_str().unwrap_or("TimeSeries").to_string();

        let data_type = meta["data_type"].as_str();
        let format = data_type
            .and_then(SampleFormat::parse)
            .ok_or_else(|| DspError::UnsupportedFormat(format!("{data_path}: data type {data_type:?}")))?;
        let shape = data.shape().to_vec();
        let (samples, channels, two_d) = match shape[..] {
            [t] => (t, 1, false),
            [t, c] => (t, c as usize, true),
            _ => return Err(DspError::UnsupportedFormat(format!("{data_path}: {}-D data", shape.len()))),
        };

        let start_meta = read_node_json(path, &format!("{series}/starting_time"))
            .ok_or_else(|| DspError::UnsupportedFormat(format!("{series} has timestamps, not a regular rate")))?;
        let rate = start_meta["attributes"]["rate"].as_f64().ok_or_else(|| err(format!("{series}/starting_time has no rate")))?;
        let start = read_all::<f64>(&store, &format!("{series}/starting_time")).and_then(|v| v.first().copied()).unwrap_or(0.0);

        // value = stored * conversion + offset, in `unit`; electrical series → µV
        let attrs = &meta["attributes"];
        let unit = attrs["unit"].as_str().unwrap_or("a.u.").to_string();
        let in_volts = kind == "ElectricalSeries" || unit == "volts";
        let to_uv = if in_volts { MICROVOLTS_PER_VOLT } else { 1.0 };
        let signal_unit = match unit.as_str() {
            _ if in_volts => SignalUnit::Microvolt,
            "" | "a.u." => SignalUnit::Dimensionless,
            other => SignalUnit::Other(other.to_string()),
        };
        let gain = attrs["conversion"].as_f64().unwrap_or(1.0) * to_uv;
        let offset = attrs["offset"].as_f64().unwrap_or(0.0) * to_uv;

        let name = format!(
            "{} · {}",
            path.file_name().map_or_else(|| "nwb".into(), |n| n.to_string_lossy().into_owned()),
            series.rsplit('/').next().unwrap_or(series)
        );
        let mut info = RecordingInfo::new(name, channels, samples, SampleRate::new(rate)?, format, MemoryOrder::TimeMajor);
        info.start_time = RationalTime::from_seconds_f64(start)?;
        for c in &mut info.channels {
            c.gain = gain as f32;
            c.offset = offset as f32;
            c.unit = signal_unit.clone();
        }
        // Channel names from the electrodes table rows this series references
        if kind == "ElectricalSeries" {
            let rows = read_all::<i64>(&store, &format!("{series}/electrodes"));
            let names = read_all::<String>(&store, "/general/extracellular_ephys/electrodes/channel_name");
            if let (Some(rows), Some(names)) = (rows, names) {
                for (c, r) in info.channels.iter_mut().zip(rows) {
                    if let Some(n) = names.get(r as usize) {
                        c.name = n.clone();
                    }
                }
            }
        }
        info.metadata.insert("format".into(), "NWB (Zarr)".into());
        info.metadata.insert("nwb_series".into(), series.to_string());
        info.metadata.insert("nwb_type".into(), kind);
        info.metadata.insert("unit".into(), if to_uv > 1.0 { "uV".into() } else { unit });
        let others: Vec<String> = list_series(path).into_iter().map(|s| s.path).filter(|p| p != series).collect();
        if !others.is_empty() {
            info.metadata.insert("nwb_other_series".into(), others.join(", "));
        }
        let origin = vec![0; data.shape().len()];
        let chunk_samples = data.chunk_shape(&origin).ok().map(|c| c[0].get());
        Ok(Self { info, data, two_d, chunk_samples })
    }

    fn retrieve(&self, ch: Range<u64>, samples: Range<u64>) -> DspResult<Vec<f32>> {
        let read = |e: zarrs::array::ArrayError| err(e);
        macro_rules! get {
            ($t:ty) => {
                if self.two_d {
                    self.data.retrieve_array_subset::<Vec<$t>>(&[samples.clone(), ch.clone()]).map_err(read)?
                } else {
                    self.data.retrieve_array_subset::<Vec<$t>>(&[samples.clone()]).map_err(read)?
                }
            };
        }
        Ok(match self.info.format {
            SampleFormat::F32 => get!(f32),
            SampleFormat::I8 => get!(i8).into_iter().map(f32::from).collect(),
            SampleFormat::I16 => get!(i16).into_iter().map(f32::from).collect(),
            SampleFormat::U16 => get!(u16).into_iter().map(f32::from).collect(),
            SampleFormat::I32 => get!(i32).into_iter().map(|v| v as f32).collect(),
            SampleFormat::F64 => get!(f64).into_iter().map(|v| v as f32).collect(),
        })
    }
}


impl RecordingSource for NwbZarrRecording {
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
            return Ok(MemoryOrder::TimeMajor);
        }
        // Stored [time, channel]: kept as is
        let block = self.retrieve(0..nch as u64, samples)?;
        out.copy_from_slice(&block);
        scale_frames(&self.info, out);
        Ok(MemoryOrder::TimeMajor)
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        let n = check_read_stored(&self.info, channels, &samples, out.len())?;
        if n == 0 || channels.is_empty() {
            return Ok(());
        }
        let lo = *channels.iter().min().unwrap();
        let hi = *channels.iter().max().unwrap() + 1;
        let bytes = self.info.format.bytes();
        let fetched = if self.two_d {
            self.data.retrieve_array_subset::<ArrayBytes<'static>>(&[samples.clone(), lo as u64..hi as u64])
        } else {
            self.data.retrieve_array_subset::<ArrayBytes<'static>>(&[samples.clone()])
        };
        let mut block = fetched.map_err(err)?.into_fixed().map_err(|e| err(e.to_string()))?.into_owned();
        native_to_le(&mut block, bytes);
        let cols: Vec<usize> = channels.iter().map(|&c| c - lo).collect();
        select_stored(&block, true, hi - lo, n, &cols, bytes, out);
        Ok(())
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        if n == 0 || channels.is_empty() {
            return Ok(());
        }
        // One [time, channel] block spanning the requested channels, then transpose
        let lo = *channels.iter().min().unwrap();
        let hi = *channels.iter().max().unwrap() + 1;
        let width = hi - lo;
        let block = self.retrieve(lo as u64..hi as u64, samples)?;
        // [time, channel] block to channel-major rows: one pass over the block, each row read once
        let selected: Vec<(usize, f32, f32)> =
            channels.iter().map(|&ch| (ch - lo, self.info.channels[ch].gain, self.info.channels[ch].offset)).collect();
        for (t, frame) in block.chunks_exact(width).enumerate() {
            for (i, &(col, gain, offset)) in selected.iter().enumerate() {
                out[i * n + t] = frame[col] * gain + offset;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use zarrs::array::{data_type, ArrayBuilder};
    use zarrs::group::GroupBuilder;
    use zarrs::storage::ReadableWritableListableStorage;

    /// A minimal NWB-Zarr store: one int16 ElectricalSeries [time, channel] with conversion,
    /// one float TimeSeries, and one timestamped series that must be ignored.
    fn write_store(path: &Path) {
        let _ = std::fs::remove_dir_all(path);
        std::fs::create_dir_all(path).unwrap();
        let s: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(path).unwrap());
        let group = |p: &str, attrs: Value| {
            let mut g = GroupBuilder::new().build(s.clone(), p).unwrap();
            *g.attributes_mut() = attrs.as_object().unwrap().clone();
            g.store_metadata().unwrap();
        };
        group("/", json!({ "neurodata_type": "NWBFile", "namespace": "core" }));
        group("/acquisition", json!({}));
        group("/acquisition/ES", json!({ "neurodata_type": "ElectricalSeries" }));
        group("/acquisition/Temp", json!({ "neurodata_type": "TimeSeries" }));
        group("/acquisition/Ticks", json!({ "neurodata_type": "TimeSeries" }));

        let arr = |p: &str, shape: Vec<u64>, dt: zarrs::array::DataType, attrs: Value| {
            let mut b = ArrayBuilder::new(shape.clone(), shape.iter().map(|&v| v.max(1)).collect::<Vec<_>>(), dt, 0i16);
            b.attributes(attrs.as_object().unwrap().clone());
            let a = b.build(s.clone(), p).unwrap();
            a.store_metadata().unwrap();
            a
        };
        // 4 samples x 3 channels, value = t * 10 + c, conversion 1e-6 V per unit → 1 µV per unit
        let es = arr("/acquisition/ES/data", vec![4, 3], data_type::int16(), json!({ "unit": "volts", "conversion": 1e-6 }));
        es.store_array_subset(&es.subset_all(), (0..4).flat_map(|t| (0..3).map(move |c| (t * 10 + c) as i16)).collect::<Vec<_>>()).unwrap();
        let mut b = ArrayBuilder::new(Vec::<u64>::new(), Vec::<u64>::new(), data_type::float64(), 0.0f64);
        b.attributes(json!({ "rate": 1000.0, "unit": "seconds" }).as_object().unwrap().clone());
        let st = b.build(s.clone(), "/acquisition/ES/starting_time").unwrap();
        st.store_metadata().unwrap();
        st.store_array_subset(&st.subset_all(), vec![0.5f64]).unwrap();

        let mut b = ArrayBuilder::new(vec![5], vec![5], data_type::float32(), 0.0f32);
        b.attributes(json!({ "unit": "a.u.", "conversion": 2.0 }).as_object().unwrap().clone());
        let temp = b.build(s.clone(), "/acquisition/Temp/data").unwrap();
        temp.store_metadata().unwrap();
        temp.store_array_subset(&temp.subset_all(), vec![1.0f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mut b = ArrayBuilder::new(Vec::<u64>::new(), Vec::<u64>::new(), data_type::float64(), 0.0f64);
        b.attributes(json!({ "rate": 10.0 }).as_object().unwrap().clone());
        let st = b.build(s.clone(), "/acquisition/Temp/starting_time").unwrap();
        st.store_metadata().unwrap();
        st.store_array_subset(&st.subset_all(), vec![0.0f64]).unwrap();

        let mut b = ArrayBuilder::new(vec![2], vec![2], data_type::float64(), 0.0f64);
        b.attributes(json!({}).as_object().unwrap().clone());
        b.build(s.clone(), "/acquisition/Ticks/data").unwrap().store_metadata().unwrap();
    }

    #[test]
    fn test_reads_series_scaled_and_transposed() {
        let path = std::env::temp_dir().join(format!("dsp_io_nwb_{}.nwb.zarr", std::process::id()));
        write_store(&path);
        assert!(is_nwb_zarr(&path));
        let series: Vec<String> = list_series(&path).into_iter().map(|s| s.path).collect();
        assert_eq!(series, vec!["/acquisition/ES", "/acquisition/Temp"], "timestamped series are skipped");

        // Default: the ElectricalSeries
        let rec = crate::open(&path).unwrap();
        let i = rec.info();
        assert_eq!((i.channel_count(), i.samples, i.sample_rate_hz(), i.start_time.as_seconds_f64()), (3, 4, 1000.0, 0.5));
        let mut out = vec![0.0; 2 * 3];
        rec.read(&[2, 0], 1..4, &mut out).unwrap();
        assert_eq!(out, vec![12.0, 22.0, 32.0, 10.0, 20.0, 30.0]);
        crate::core::tests::assert_stored_matches(rec.as_ref(), &[2, 0], 1..4);
        crate::core::tests::assert_native_matches(rec.as_ref(), 1..4);

        let listed: Vec<(String, dsp_core::SignalUnit, f64, f64)> = crate::sources(&path)
            .unwrap()
            .into_iter()
            .map(|s| (s.name, s.unit, s.sample_rate.rate_hz(), s.start_time.as_seconds_f64()))
            .collect();
        assert_eq!(
            listed,
            vec![("ES".into(), dsp_core::SignalUnit::Microvolt, 1000.0, 0.5), ("Temp".into(), dsp_core::SignalUnit::Dimensionless, 10.0, 0.0)]
        );
        assert_eq!(crate::open_source(&path, "/acquisition/Temp").unwrap().info().channel_count(), 1);

        let temp = NwbZarrRecording::open_series(&path, "/acquisition/Temp").unwrap();
        let mut out = vec![0.0; 2];
        temp.read(&[0], 3..5, &mut out).unwrap();
        assert_eq!(out, vec![8.0, 10.0]);
        std::fs::remove_dir_all(&path).unwrap();
    }

    #[test]
    fn test_reads_int8_int32_and_float64_series() {
        let path = std::env::temp_dir().join(format!("dsp_io_nwb_types_{}.nwb.zarr", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        let s: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&path).unwrap());
        let group = |p: &str, attrs: Value| {
            let mut g = GroupBuilder::new().build(s.clone(), p).unwrap();
            *g.attributes_mut() = attrs.as_object().unwrap().clone();
            g.store_metadata().unwrap();
        };
        group("/", json!({ "neurodata_type": "NWBFile", "namespace": "core" }));
        group("/acquisition", json!({}));
        // 4 samples x 3 channels, value = t * 10 − c (negative values exercise signed types)
        let values: Vec<i32> = (0..4).flat_map(|t| (0..3).map(move |c| t * 10 - c)).collect();
        macro_rules! series {
            ($name:expr, $dt:expr, $t:ty) => {{
                group(&format!("/acquisition/{}", $name), json!({ "neurodata_type": "TimeSeries" }));
                let b = ArrayBuilder::new(vec![4u64, 3], vec![4u64, 3], $dt, 0 as $t);
                let a = b.build(s.clone(), &format!("/acquisition/{}/data", $name)).unwrap();
                a.store_metadata().unwrap();
                a.store_array_subset(&a.subset_all(), values.iter().map(|&v| v as $t).collect::<Vec<$t>>()).unwrap();
                let mut b = ArrayBuilder::new(Vec::<u64>::new(), Vec::<u64>::new(), data_type::float64(), 0.0f64);
                b.attributes(json!({ "rate": 100.0 }).as_object().unwrap().clone());
                let st = b.build(s.clone(), &format!("/acquisition/{}/starting_time", $name)).unwrap();
                st.store_metadata().unwrap();
                st.store_array_subset(&st.subset_all(), vec![0.0f64]).unwrap();
            }};
        }
        series!("I8", data_type::int8(), i8);
        series!("I32", data_type::int32(), i32);
        series!("F64", data_type::float64(), f64);

        for (name, format) in [("I8", SampleFormat::I8), ("I32", SampleFormat::I32), ("F64", SampleFormat::F64)] {
            let rec = NwbZarrRecording::open_series(&path, &format!("/acquisition/{name}")).unwrap();
            assert_eq!(rec.info().format, format);
            let mut out = vec![0.0; 2 * 3];
            rec.read(&[2, 0], 1..4, &mut out).unwrap();
            assert_eq!(out, vec![8.0, 18.0, 28.0, 10.0, 20.0, 30.0], "{name}");
            crate::core::tests::assert_stored_matches(&rec, &[2, 0], 1..4);
        }
        std::fs::remove_dir_all(&path).unwrap();
    }
}
