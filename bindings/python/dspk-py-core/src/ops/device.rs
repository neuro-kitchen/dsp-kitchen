//! Running one operation on the device for a NumPy array: open the runtime's client, upload,
//! run, download. Shared by the one-shot functions of `dspk.ops`.

use cubecl::prelude::Client;
use cubecl::server::Handle;
use dspk_ops::core::buffer;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::runtime;

/// Uploads `data` `(channels, samples)`, runs `op(client, input, output, channels, samples)` on the
/// runtime `runtime` selects (see `dspk.runtime`) with the GIL released, and returns the output.
/// `op` writes `channels · samples` values to `output`; an `Err` becomes `RuntimeError`.
pub(crate) fn run<'py, Op>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, runtime: Option<&str>, op: Op) -> PyResult<Bound<'py, PyArray2<f32>>>
where
    Op: FnOnce(&Client, &Handle, &Handle, usize, usize) -> Result<(), String> + Send,
{
    let target = runtime::target(runtime)?;
    let (channels, samples) = data.as_array().dim();
    let values: Vec<f32> = data.as_array().iter().copied().collect();
    let out = py
        .detach(move || -> Result<Vec<f32>, String> {
            let client = runtime::open_client(target)?;
            let input = buffer::upload(&client, &values);
            let output = buffer::empty::<f32>(&client, values.len());
            op(&client, &input, &output, channels, samples)?;
            Ok(buffer::download_prefix::<f32>(&client, output, values.len()))
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(Array2::from_shape_vec((channels, samples), out).expect("one value per input value").into_pyarray(py))
}

/// Like [`run`] for operations that work in place: `op(client, buffer, channels, samples)` changes
/// the uploaded `data` itself, which is then downloaded.
pub(crate) fn run_in_place<'py, Op>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, runtime: Option<&str>, op: Op) -> PyResult<Bound<'py, PyArray2<f32>>>
where
    Op: FnOnce(&Client, &Handle, usize, usize) -> Result<(), String> + Send,
{
    let target = runtime::target(runtime)?;
    let (channels, samples) = data.as_array().dim();
    let values: Vec<f32> = data.as_array().iter().copied().collect();
    let out = py
        .detach(move || -> Result<Vec<f32>, String> {
            let client = runtime::open_client(target)?;
            let signal = buffer::upload(&client, &values);
            op(&client, &signal, channels, samples)?;
            Ok(buffer::download_prefix::<f32>(&client, signal, values.len()))
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(Array2::from_shape_vec((channels, samples), out).expect("one value per input value").into_pyarray(py))
}
