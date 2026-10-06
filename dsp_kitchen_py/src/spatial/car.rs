//! Common average reference: each sample minus the mean of all channels at that sample.

use dsp_base::pipeline::PipelineStage;
use pyo3::prelude::*;

use crate::pipeline::run_stage;

#[pyclass(name = "CommonAverageReference", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCommonAverageReference;

#[pymethods]
impl PyCommonAverageReference {
    #[new]
    fn new() -> Self {
        Self
    }
    fn __repr__(&self) -> String {
        "CommonAverageReference()".to_string()
    }
}

/// Common average reference of `data` (`[channels, samples]`) on the device.
#[pyfunction]
#[pyo3(signature = (data, *, runtime=None))]
fn common_average_reference<'py>(py: Python<'py>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PipelineStage::CommonAverageReference, &data, None, runtime)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCommonAverageReference>()?;
    m.add_function(wrap_pyfunction!(common_average_reference, m)?)?;
    Ok(())
}
