//! Common average reference: each sample minus the mean of all channels at that sample.

use dsp_base::pipeline::PipelineStage;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::pipeline::run_stage;

/// Common average reference: a pipeline stage subtracting, at every sample, the mean over channels.
///
/// Removes signals shared by all channels (reference noise, far-field activity). Leave it out when real
/// signals span most channels (muscle arrays).
///
/// Examples
/// --------
/// >>> y = Pipeline([CommonAverageReference(), HighpassFilter(300.0)]).run(x, fs=30000.0)
#[gen_stub_pyclass]
#[pyclass(name = "CommonAverageReference", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCommonAverageReference;

#[gen_stub_pymethods]
#[pymethods]
impl PyCommonAverageReference {
    /// See the class docs.
    #[new]
    fn new() -> Self {
        Self
    }
    fn __repr__(&self) -> String {
        "CommonAverageReference()".to_string()
    }
}

/// Common average reference of an array: every sample minus the mean over channels.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]`, converted to float32.
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// numpy.ndarray
///     Float32, same shape as `data`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (data, *, runtime=None))]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn common_average_reference<'py>(py: Python<'py>, data: Bound<'py, PyAny>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PipelineStage::CommonAverageReference, &data, None, runtime)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCommonAverageReference>()?;
    m.add_function(wrap_pyfunction!(common_average_reference, m)?)?;
    Ok(())
}
