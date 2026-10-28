//! `dspk.io`: open recordings and read them into NumPy; write NumPy arrays as raw binary or Zarr.
//!
//! These are the generic formats of `dspk-io` (raw binary with a JSON sidecar, Zarr `/traces`);
//! neural formats (NWB, SpikeGLX, …) come with the synapse bindings.

use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use dspk_core::{MemoryOrder, RecordingInfo, RecordingSource, SampleFormat, SampleRate, SignalUnit};
use dspk_core::recording::check_read;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::to_py;

/// Samples per channel each writer step reads from the source (bounds memory while writing).
pub const WRITE_CHUNK_SAMPLES: usize = 65_536;

/// A recording opened from a file. Reads return scaled `float32` values in each channel's unit,
/// shape `(channels, samples)`.
#[pyclass(module = "dspk.io", frozen)]
pub struct Recording {
    inner: Arc<dyn RecordingSource>,
}

#[pymethods]
impl Recording {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.info().name
    }

    /// Number of channels.
    #[getter]
    fn channel_count(&self) -> usize {
        self.inner.info().channel_count()
    }

    /// Samples per channel.
    #[getter]
    fn samples(&self) -> u64 {
        self.inner.info().samples
    }

    /// Sampling rate in Hz.
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.info().sample_rate_hz()
    }

    /// Duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.inner.info().duration_sec()
    }

    /// Stored sample type (`"int16"`, `"float32"`, …).
    #[getter]
    fn format(&self) -> &'static str {
        self.inner.info().format.name()
    }

    #[getter]
    fn channel_names(&self) -> Vec<String> {
        self.inner.info().channels.iter().map(|c| c.name.clone()).collect()
    }

    /// Unit symbol of each channel's values (`"µV"`, `"mV"`, …; `""` when dimensionless).
    #[getter]
    fn units(&self) -> Vec<String> {
        self.inner.info().channels.iter().map(|c| c.unit.symbol().to_string()).collect()
    }

    /// Reads `channels` (default: all) over samples `start..stop` (default: to the end).
    #[pyo3(signature = (channels=None, start=0, stop=None))]
    fn read<'py>(&self, py: Python<'py>, channels: Option<Vec<usize>>, start: u64, stop: Option<u64>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let info = self.inner.info();
        let channels = channels.unwrap_or_else(|| (0..info.channel_count()).collect());
        let samples = start..stop.unwrap_or(info.samples);
        let n = samples.end.saturating_sub(samples.start) as usize;
        let source = Arc::clone(&self.inner);
        let values = py
            .detach(move || {
                let mut out = vec![0.0f32; channels.len() * n];
                source.read(&channels, samples, &mut out).map(|()| (channels.len(), out))
            })
            .map_err(to_py)?;
        let (rows, out) = values;
        Ok(Array2::from_shape_vec((rows, n), out).expect("read fills rows × samples").into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        let info = self.inner.info();
        format!(
            "Recording(name={:?}, channels={}, samples={}, sample_rate={} Hz, format={})",
            info.name,
            info.channel_count(),
            info.samples,
            info.sample_rate_hz(),
            info.format.name()
        )
    }
}

/// One source of a file (most files have one; containers such as NWB have several).
#[pyclass(module = "dspk.io", frozen, get_all)]
pub struct Source {
    /// Id to pass to `open(path, source=...)`.
    id: String,
    name: String,
    channel_count: usize,
    samples: u64,
    /// Sampling rate in Hz.
    sample_rate: f64,
    format: &'static str,
    /// Unit symbol of the values (`""` when dimensionless).
    unit: String,
}

#[pymethods]
impl Source {
    fn __repr__(&self) -> String {
        format!("Source(id={:?}, channels={}, samples={}, sample_rate={} Hz)", self.id, self.channel_count, self.samples, self.sample_rate)
    }
}

/// Opens the recording at `path`: its default source, or `source` (an id from `sources(path)`).
#[pyfunction]
#[pyo3(signature = (path, source=None))]
pub fn open(path: PathBuf, source: Option<&str>) -> PyResult<Recording> {
    let inner = match source {
        Some(id) => dspk_io::open_source(&path, id),
        None => dspk_io::open(&path),
    }
    .map_err(to_py)?;
    Ok(Recording { inner: Arc::from(inner) })
}

/// Every source of the file at `path`, from its metadata only.
#[pyfunction]
pub fn sources(path: PathBuf) -> PyResult<Vec<Source>> {
    Ok(dspk_io::sources(&path)
        .map_err(to_py)?
        .into_iter()
        .map(|e| Source {
            id: e.id,
            name: e.name,
            channel_count: e.channels,
            samples: e.samples,
            sample_rate: e.sample_rate.rate_hz(),
            format: e.format.name(),
            unit: e.unit.symbol().to_string(),
        })
        .collect())
}

/// Names of the formats `open` recognizes, in detection order.
#[pyfunction]
pub fn formats() -> Vec<&'static str> {
    dspk_io::formats().iter().map(|f| f.name()).collect()
}

/// A NumPy array `(channels, samples)` seen as a recording, so the Rust writers can read it.
struct ArrayRecording {
    info: RecordingInfo,
    data: Vec<f32>,
}

impl ArrayRecording {
    fn new(data: PyReadonlyArray2<'_, f32>, sample_rate: f64, unit: &str) -> PyResult<Self> {
        let view = data.as_array();
        let (channels, samples) = view.dim();
        if channels == 0 {
            return Err(PyValueError::new_err("data has no channels (expected shape (channels, samples))"));
        }
        let info = RecordingInfo::new("array", channels, samples as u64, SampleRate::new(sample_rate).map_err(to_py)?, SampleFormat::F32, MemoryOrder::ChannelMajor)
            .with_gain(1.0, parse_unit(unit));
        Ok(Self { info, data: view.iter().copied().collect() })
    }
}

impl RecordingSource for ArrayRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> dspk_core::DspResult<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        let total = self.info.samples as usize;
        for (dst, &ch) in out.chunks_exact_mut(n.max(1)).zip(channels) {
            let base = ch * total + samples.start as usize;
            dst.copy_from_slice(&self.data[base..base + n]);
        }
        Ok(())
    }
}

/// `""` (dimensionless), `"V"`, `"mV"`, `"µV"` / `"uV"`, `"A"`, `"mA"`, `"µA"` / `"uA"`, or any other
/// symbol, kept as given.
fn parse_unit(symbol: &str) -> SignalUnit {
    match symbol.trim() {
        "" => SignalUnit::Dimensionless,
        "V" => SignalUnit::Volt,
        "mV" => SignalUnit::Millivolt,
        "µV" | "uV" => SignalUnit::Microvolt,
        "A" => SignalUnit::Ampere,
        "mA" => SignalUnit::Milliampere,
        "µA" | "uA" => SignalUnit::Microampere,
        other => SignalUnit::Other(other.to_string()),
    }
}

fn parse_format(name: &str) -> PyResult<SampleFormat> {
    SampleFormat::parse(name).ok_or_else(|| PyValueError::new_err(format!("unknown sample format {name:?} (int8, int16, uint16, int32, float32, float64)")))
}

fn parse_order(name: &str) -> PyResult<MemoryOrder> {
    match name {
        "channel_major" => Ok(MemoryOrder::ChannelMajor),
        "time_major" => Ok(MemoryOrder::TimeMajor),
        other => Err(PyValueError::new_err(format!("unknown order {other:?} (channel_major or time_major)"))),
    }
}

/// Writes `data` `(channels, samples)` as raw binary at `path` with its JSON sidecar.
///
/// `format`: stored type; values are stored as `round(value / gain)` for integer formats.
/// `order`: `"channel_major"` or `"time_major"`. `unit`: symbol of the values (`""`: dimensionless).
#[pyfunction]
#[pyo3(signature = (data, path, sample_rate, *, format="float32", order="channel_major", gain=1.0, unit="", chunk_samples=WRITE_CHUNK_SAMPLES))]
#[allow(clippy::too_many_arguments)]
pub fn write_raw(py: Python<'_>, data: PyReadonlyArray2<'_, f32>, path: PathBuf, sample_rate: f64, format: &str, order: &str, gain: f32, unit: &str, chunk_samples: usize) -> PyResult<()> {
    let (format, order) = (parse_format(format)?, parse_order(order)?);
    let source = ArrayRecording::new(data, sample_rate, unit)?;
    py.detach(|| dspk_io::write_raw(&source, &path, format, order, gain, chunk_samples, |_, _| {})).map_err(to_py)?;
    Ok(())
}

/// Writes `data` `(channels, samples)` as a Zarr store at `path` (`/traces`, `float32`).
#[pyfunction]
#[pyo3(signature = (data, path, sample_rate, *, unit="", chunk_samples=WRITE_CHUNK_SAMPLES))]
pub fn write_zarr(py: Python<'_>, data: PyReadonlyArray2<'_, f32>, path: PathBuf, sample_rate: f64, unit: &str, chunk_samples: usize) -> PyResult<()> {
    let source = ArrayRecording::new(data, sample_rate, unit)?;
    py.detach(|| dspk_io::write_zarr(&source, &path, chunk_samples, |_, _| {})).map_err(to_py)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Recording>()?;
    m.add_class::<Source>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(sources, m)?)?;
    m.add_function(wrap_pyfunction!(formats, m)?)?;
    m.add_function(wrap_pyfunction!(write_raw, m)?)?;
    m.add_function(wrap_pyfunction!(write_zarr, m)?)?;
    m.add("WRITE_CHUNK_SAMPLES", WRITE_CHUNK_SAMPLES)?;
    Ok(())
}
