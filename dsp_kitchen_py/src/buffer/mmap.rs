use std::path::Path;
use std::sync::Arc;

use dsp_core::{MemoryOrder, RecordingSource, SampleFormat};
use dsp_io::{RawParams, RawRecording};
use numpy::ndarray::{ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{Element, PyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use pyo3::types::IntoPyDict;

use super::recording::read_to_numpy;

/// Bytes in a MiB (for `repr`).
const BYTES_PER_MIB: f64 = 1_048_576.0;

/// A raw binary recording, memory-mapped (dsp-io `RawRecording`).
///
/// The layout comes from the JSON sidecar next to the file (`rec.bin` → `rec.meta`) or from the
/// arguments. `read()` returns float32 copies in µV (gain and offset applied); `to_numpy()` is a
/// zero-copy, read-only view of the stored samples that keeps this recording (and its mapping) alive.
///
/// Examples
/// --------
/// >>> from dsp_kitchen.io import MmapRecording
/// >>> rec = MmapRecording("data.bin", 384, 30000.0, dtype="int16", gain=2.34)
/// >>> raw = rec.to_numpy()          # [channels, samples] int16, no copy
/// >>> x = rec.read(0, 30000)        # [channels, samples] float32, µV
#[gen_stub_pyclass]
#[pyclass(name = "MmapRecording", frozen, skip_from_py_object)]
pub struct PyMmapRecording {
    path: String,
    inner: Arc<RawRecording>,
}

fn parse_order(order: &str) -> PyResult<MemoryOrder> {
    match order {
        "channel_major" => Ok(MemoryOrder::ChannelMajor),
        "time_major" => Ok(MemoryOrder::TimeMajor),
        other => Err(PyValueError::new_err(format!(
            "order must be 'channel_major' or 'time_major', got '{other}'"
        ))),
    }
}

impl PyMmapRecording {
    /// Zero-copy `(channels, samples)` view of the stored values with this object as the base.
    fn stored_view<'py, T: Element>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let this = slf.get();
        let info = this.inner.info();
        let (nch, ns) = (info.channels.len(), info.samples as usize);
        let bytes = this.inner.stored_bytes();
        if !bytes.as_ptr().cast::<T>().is_aligned() {
            return Err(PyValueError::new_err(
                "stored samples are not aligned for a zero-copy view (header_bytes); use read()",
            ));
        }
        // SAFETY: the bytes hold `nch * ns` values of T (checked by RawRecording::open_with) and are
        // aligned (checked above); they live in the mapping owned by `slf`, which becomes the
        // array's base object and is immutable (`frozen`).
        let values: &[T] = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), nch * ns) };
        let view = match info.order {
            MemoryOrder::ChannelMajor => ArrayView2::from_shape((nch, ns), values),
            MemoryOrder::TimeMajor => ArrayView2::from_shape((nch, ns).strides((1, nch)), values),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let py = slf.py();
        let arr = unsafe { PyArray::borrow_from_array(&view, slf.clone().into_any()) };
        arr.call_method("setflags", (), Some(&[("write", false)].into_py_dict(py)?))?;
        Ok(arr.into_any())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMmapRecording {
    /// Opens a raw binary file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    /// channels : int, optional
    ///     Channels; with `sample_rate`, describes the layout. Without them, the JSON sidecar next to the
    ///     file (`rec.bin` → `rec.meta`) does.
    /// sample_rate : float, optional
    ///     Hz.
    /// dtype : {"int8", "int16", "uint16", "int32", "float32", "float64"}, default "float32"
    ///     Stored sample type (NumPy spellings such as `"<i2"` also work).
    /// order : {"channel_major", "time_major"}, default "channel_major"
    ///     `"channel_major"`: all samples of channel 0, then channel 1, …; `"time_major"`: interleaved.
    /// gain : float, default 1.0
    ///     µV per stored step.
    /// offset : float, default 0.0
    ///     µV added after the gain.
    /// header_bytes : int, default 0
    ///     Bytes before the first sample.
    /// samples : int, optional
    ///     Samples per channel; default: from the file size.
    #[new]
    #[pyo3(signature = (path, channels=None, sample_rate=None, *, dtype="float32", order="channel_major", gain=1.0, offset=0.0, header_bytes=0, samples=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: &str,
        channels: Option<usize>,
        sample_rate: Option<f64>,
        dtype: &str,
        order: &str,
        gain: f32,
        offset: f32,
        header_bytes: u64,
        samples: Option<u64>,
    ) -> PyResult<Self> {
        let p = Path::new(path);
        let inner = match (channels, sample_rate) {
            (None, None) => RawRecording::open(p),
            (Some(channels), Some(rate)) => {
                let format = SampleFormat::parse(dtype)
                    .ok_or_else(|| PyValueError::new_err(format!("unsupported dtype '{dtype}'")))?;
                let mut params = RawParams::new(channels, rate, format, parse_order(order)?);
                params.gain_uv = gain;
                params.offset_uv = offset;
                params.header_bytes = header_bytes;
                params.samples = samples;
                RawRecording::open_with(p, &params)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "give both channels and sample_rate, or neither to use the JSON sidecar",
                ));
            }
        }
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))?;
        Ok(Self { path: path.to_string(), inner: Arc::new(inner) })
    }

    /// Path of the file.
    #[getter]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Number of channels.
    #[getter]
    pub fn channels(&self) -> usize {
        self.inner.info().channels.len()
    }

    /// Samples per channel.
    #[getter]
    pub fn samples(&self) -> u64 {
        self.inner.info().samples
    }

    /// Sampling rate, Hz.
    #[getter]
    pub fn sample_rate(&self) -> f64 {
        self.inner.info().sample_rate_hz()
    }

    /// `(channels, samples)`.
    #[getter]
    pub fn shape(&self) -> (usize, u64) {
        (self.channels(), self.samples())
    }

    /// NumPy dtype name of the stored samples.
    #[getter]
    pub fn dtype(&self) -> &'static str {
        self.inner.info().format.name()
    }

    /// Size of the stored samples, bytes.
    #[getter]
    pub fn total_bytes(&self) -> usize {
        self.inner.stored_bytes().len()
    }

    /// Reads samples `[start_sample, end_sample)` (default: one second) as `[channels, samples]` float32
    /// in µV (gain and offset applied; a copy).
    ///
    /// Parameters
    /// ----------
    /// start_sample : int, default 0
    /// end_sample : int, optional
    /// channels : list of int, optional
    ///     Channel indices; default: all.
    #[pyo3(signature = (start_sample=0, end_sample=None, channels=None))]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
    pub fn read<'py>(
        &self,
        py: Python<'py>,
        start_sample: u64,
        end_sample: Option<u64>,
        channels: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        read_to_numpy(py, self.inner.as_ref(), start_sample, end_sample, channels)
    }

    /// Zero-copy read-only view `[channels, samples]` of the stored values (stored dtype, before
    /// gain). Keeps this recording alive while the array exists.
    pub fn to_numpy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        match slf.get().inner.info().format {
            SampleFormat::I8 => Self::stored_view::<i8>(slf),
            SampleFormat::I16 => Self::stored_view::<i16>(slf),
            SampleFormat::U16 => Self::stored_view::<u16>(slf),
            SampleFormat::I32 => Self::stored_view::<i32>(slf),
            SampleFormat::F32 => Self::stored_view::<f32>(slf),
            SampleFormat::F64 => Self::stored_view::<f64>(slf),
        }
    }

    /// Zero-copy read-only memoryview of the stored bytes. Keeps this recording alive.
    pub fn memoryview<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let bytes = slf.get().inner.stored_bytes();
        let view = ArrayView1::from(bytes);
        // SAFETY: as in `stored_view`: the bytes live in the mapping owned by the base object.
        let arr = unsafe { PyArray::borrow_from_array(&view, slf.clone().into_any()) };
        let py = slf.py();
        arr.call_method("setflags", (), Some(&[("write", false)].into_py_dict(py)?))?;
        py.import("builtins")?.getattr("memoryview")?.call1((arr,))
    }

    fn __repr__(&self) -> String {
        format!(
            "MmapRecording(path='{}', shape=({}, {}), dtype={}, sample_rate={:.1}Hz, size={:.2} MiB)",
            self.path,
            self.channels(),
            self.samples(),
            self.dtype(),
            self.sample_rate(),
            self.total_bytes() as f64 / BYTES_PER_MIB
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMmapRecording>()
}
