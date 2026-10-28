//! `dspk.ops.spatial`: operations across channels on the device (`dspk_ops::spatial`).

use dspk_ops::spatial::{execute_common_median, execute_direct_car};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::device;

/// Common average reference: the mean over channels subtracted at every sample.
#[gen_stub_pyfunction(module = "dspk._native.ops.spatial")]
#[pyfunction]
#[pyo3(signature = (data, *, runtime))]
pub fn car<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    device::run(py, data, runtime, |client, input, output, channels, samples| {
        execute_direct_car::<f32>(client, input, output, channels, samples);
        Ok(())
    })
}

/// Common median reference: the median over channels subtracted at every sample.
#[gen_stub_pyfunction(module = "dspk._native.ops.spatial")]
#[pyfunction]
#[pyo3(signature = (data, *, runtime))]
pub fn cmr<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    device::run(py, data, runtime, |client, input, output, channels, samples| {
        execute_common_median::<f32>(client, input, output, channels, samples);
        Ok(())
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(car, m)?)?;
    m.add_function(wrap_pyfunction!(cmr, m)?)?;
    Ok(())
}
