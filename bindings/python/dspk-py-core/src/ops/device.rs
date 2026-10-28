//! Running one operation on the device for a NumPy array: open the runtime's client, upload,
//! run, download. Shared by the one-shot functions of `dspk.ops`.
//!
//! Inputs cross in their own dtype ([`DeviceInput`]): float32 as is, integer arrays as their
//! stored integers, converted to float32 on the device. Results are float32.

use cubecl::prelude::Client;
use cubecl::server::Handle;
use dspk_core::SampleFormat;
use dspk_ops::core::buffer;
use dspk_ops::math::{execute_unpack_stored, upload_stored};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::{PyStubType, TypeInfo};

use crate::runtime;

/// A `(channels, samples)` NumPy array, kept in its dtype until it reaches the device.
pub struct DeviceInput {
    channels: usize,
    samples: usize,
    values: Values,
}

enum Values {
    F32(Vec<f32>),
    /// Little-endian integers of `format`, channel-major; the value is the integer.
    Stored(SampleFormat, Vec<u8>),
}

impl DeviceInput {
    pub(crate) fn dim(&self) -> (usize, usize) {
        (self.channels, self.samples)
    }

    pub(crate) fn len(&self) -> usize {
        self.channels * self.samples
    }

    /// The values as float32 on the host, for the fits whose dspk-ops entry points take host data.
    pub(crate) fn host_f32(&self) -> Vec<f32> {
        match &self.values {
            Values::F32(v) => v.clone(),
            Values::Stored(format, bytes) => {
                let mut out = vec![0.0f32; self.len()];
                format.decode(bytes, &mut out, 1.0, 0.0);
                out
            }
        }
    }

    /// The stored format and bytes of an integer array (`None` for float32).
    pub(crate) fn stored(&self) -> Option<(SampleFormat, &[u8])> {
        match &self.values {
            Values::F32(_) => None,
            Values::Stored(format, bytes) => Some((*format, bytes)),
        }
    }

    /// Uploads into a new float32 device buffer `[channels, samples]`: float32 as is, integers as
    /// their stored words, converted on the device.
    pub(crate) fn upload(&self, client: &Client) -> Result<Handle, String> {
        match &self.values {
            Values::F32(v) => Ok(buffer::upload(client, v)),
            Values::Stored(format, bytes) => {
                let words = upload_stored(client, bytes);
                let gains = buffer::upload(client, &vec![1.0f32; self.channels]);
                let offsets = buffer::upload(client, &vec![0.0f32; self.channels]);
                let out = buffer::empty::<f32>(client, self.len());
                execute_unpack_stored::<f32>(client, &words, *format, &gains, &offsets, &out, self.channels, self.samples)
                    .map_err(|e| e.to_string())?;
                Ok(out)
            }
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for DeviceInput {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        macro_rules! stored {
            ($t:ty, $format:expr) => {
                if let Ok(a) = obj.extract::<PyReadonlyArray2<'_, $t>>() {
                    let view = a.as_array();
                    let (channels, samples) = view.dim();
                    let bytes = view.iter().flat_map(|v| v.to_le_bytes()).collect();
                    return Ok(Self { channels, samples, values: Values::Stored($format, bytes) });
                }
            };
        }
        if let Ok(a) = obj.extract::<PyReadonlyArray2<'_, f32>>() {
            let view = a.as_array();
            let (channels, samples) = view.dim();
            return Ok(Self { channels, samples, values: Values::F32(view.iter().copied().collect()) });
        }
        stored!(i8, SampleFormat::I8);
        stored!(i16, SampleFormat::I16);
        stored!(u16, SampleFormat::U16);
        stored!(i32, SampleFormat::I32);
        Err(PyValueError::new_err("data must be a 2-D float32, int8, int16, uint16 or int32 array (channels, samples)"))
    }
}

impl PyStubType for DeviceInput {
    fn type_output() -> TypeInfo {
        PyUntypedArray::type_output()
    }
}

/// Uploads `data` `(channels, samples)`, runs `op(client, input, output, channels, samples)` on the
/// runtime `runtime` selects (see `dspk.runtime`) with the GIL released, and returns the output.
/// `op` writes `channels · samples` values to `output`; an `Err` becomes `RuntimeError`.
pub(crate) fn run<'py, Op>(py: Python<'py>, data: DeviceInput, runtime: Option<&str>, op: Op) -> PyResult<Bound<'py, PyArray2<f32>>>
where
    Op: FnOnce(&Client, &Handle, &Handle, usize, usize) -> Result<(), String> + Send,
{
    let samples = data.dim().1;
    run_resized(py, data, runtime, samples, op)
}

/// [`run`] for operations whose output has `out_samples` per channel (resampling).
pub(crate) fn run_resized<'py, Op>(
    py: Python<'py>,
    data: DeviceInput,
    runtime: Option<&str>,
    out_samples: usize,
    op: Op,
) -> PyResult<Bound<'py, PyArray2<f32>>>
where
    Op: FnOnce(&Client, &Handle, &Handle, usize, usize) -> Result<(), String> + Send,
{
    let channels = data.dim().0;
    run_shaped(py, data, runtime, (channels, out_samples), op)
}

/// [`run`] for operations whose output is `out_shape` `(rows, columns)` (projections, resampling).
pub(crate) fn run_shaped<'py, Op>(
    py: Python<'py>,
    data: DeviceInput,
    runtime: Option<&str>,
    out_shape: (usize, usize),
    op: Op,
) -> PyResult<Bound<'py, PyArray2<f32>>>
where
    Op: FnOnce(&Client, &Handle, &Handle, usize, usize) -> Result<(), String> + Send,
{
    let target = runtime::target(runtime)?;
    let (channels, samples) = data.dim();
    let out_len = out_shape.0 * out_shape.1;
    let out = py
        .detach(move || -> Result<Vec<f32>, String> {
            let client = runtime::open_client(target)?;
            let input = data.upload(&client)?;
            let output = buffer::empty::<f32>(&client, out_len);
            op(&client, &input, &output, channels, samples)?;
            Ok(buffer::download_prefix::<f32>(&client, output, out_len))
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(Array2::from_shape_vec(out_shape, out).expect("one value per output value").into_pyarray(py))
}

/// Uploads `data` and runs `op(client, input, channels, samples)` for a result that is not an
/// array of the same layout (statistics, peak tables).
pub(crate) fn with_input<T, Op>(py: Python<'_>, data: DeviceInput, runtime: Option<&str>, op: Op) -> PyResult<T>
where
    T: Send,
    Op: FnOnce(&Client, &Handle, usize, usize) -> Result<T, String> + Send,
{
    let target = runtime::target(runtime)?;
    let (channels, samples) = data.dim();
    py.detach(move || -> Result<T, String> {
        let client = runtime::open_client(target)?;
        let input = data.upload(&client)?;
        op(&client, &input, channels, samples)
    })
    .map_err(PyRuntimeError::new_err)
}

/// Like [`run`] for operations that work in place: `op(client, buffer, channels, samples)` changes
/// the uploaded `data` itself, which is then downloaded.
pub(crate) fn run_in_place<'py, Op>(py: Python<'py>, data: DeviceInput, runtime: Option<&str>, op: Op) -> PyResult<Bound<'py, PyArray2<f32>>>
where
    Op: FnOnce(&Client, &Handle, usize, usize) -> Result<(), String> + Send,
{
    let target = runtime::target(runtime)?;
    let (channels, samples) = data.dim();
    let out = py
        .detach(move || -> Result<Vec<f32>, String> {
            let client = runtime::open_client(target)?;
            let signal = data.upload(&client)?;
            op(&client, &signal, channels, samples)?;
            Ok(buffer::download_prefix::<f32>(&client, signal, channels * samples))
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(Array2::from_shape_vec((channels, samples), out).expect("one value per input value").into_pyarray(py))
}
