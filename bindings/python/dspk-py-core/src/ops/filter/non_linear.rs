//! Median and Teager–Kaiser on the device (`dspk_ops::filter::non_linear`).

use dspk_ops::filter::non_linear::{MAX_MEDIAN_WIDTH, MEDIAN_DEFAULT_EDGE, TEAGER_KAISER_DEFAULT_EDGE};
use dspk_ops::filter::{execute_median, execute_teager_kaiser};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::{edge, edge_name};
use crate::ops::device;

/// Running median over an odd `width` (≤ `MAX_MEDIAN_WIDTH`) along time; with `edge="zeros"` this
/// is `scipy.signal.medfilt`.
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
#[pyo3(signature = (data, width, *, edge, runtime))]
pub fn median<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, width: usize, edge: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if width.is_multiple_of(2) || width > MAX_MEDIAN_WIDTH {
        return Err(PyValueError::new_err(format!("median width must be odd and at most {MAX_MEDIAN_WIDTH}, got {width}")));
    }
    let edge = self::edge(edge)?;
    device::run(py, data, runtime, move |client, input, output, channels, samples| {
        execute_median::<f32>(client, input, output, channels, samples, width, edge);
        Ok(())
    })
}

/// Teager–Kaiser energy `x[t]² − x[t−1] · x[t+1]` along time; the neighbours of the end samples
/// come from `edge`.
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
#[pyo3(signature = (data, *, edge, runtime))]
pub fn teager_kaiser<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, edge: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let edge = self::edge(edge)?;
    device::run(py, data, runtime, move |client, input, output, channels, samples| {
        execute_teager_kaiser::<f32>(client, input, output, channels, samples, edge);
        Ok(())
    })
}

pyo3_stub_gen::module_variable!("dspk._native.ops.filter", "MEDIAN_DEFAULT_EDGE", String);
pyo3_stub_gen::module_variable!("dspk._native.ops.filter", "TEAGER_KAISER_DEFAULT_EDGE", String);
pyo3_stub_gen::module_variable!("dspk._native.ops.filter", "MAX_MEDIAN_WIDTH", usize);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(teager_kaiser, m)?)?;
    m.add("MEDIAN_DEFAULT_EDGE", edge_name(MEDIAN_DEFAULT_EDGE))?;
    m.add("TEAGER_KAISER_DEFAULT_EDGE", edge_name(TEAGER_KAISER_DEFAULT_EDGE))?;
    m.add("MAX_MEDIAN_WIDTH", MAX_MEDIAN_WIDTH)?;
    Ok(())
}
