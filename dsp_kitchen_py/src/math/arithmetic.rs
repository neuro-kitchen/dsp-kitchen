//! `dsp_kitchen.math`: pointwise stages (scale, baseline, clamp). No defaults: the values depend
//! on the recording.

use dsp_base::pipeline::PipelineStage;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::pipeline::run_stage;

/// Pointwise `x · alpha + beta`: a pipeline stage (e.g. stored integer steps to physical units).
///
/// Parameters
/// ----------
/// alpha : float
///     Gain.
/// beta : float, default 0.0
///     Offset, added after the gain.
#[gen_stub_pyclass]
#[pyclass(name = "Scale", skip_from_py_object)]
#[derive(Clone)]
pub struct PyScale {
    pub stage: PipelineStage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyScale {
    /// See the class docs.
    #[new]
    #[pyo3(signature = (alpha, beta=0.0))]
    fn new(alpha: f32, beta: f32) -> Self {
        Self { stage: PipelineStage::Scale { alpha, beta } }
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.stage)
    }
}

/// Pointwise `x − baseline`: a pipeline stage.
///
/// Parameters
/// ----------
/// baseline : float
///     Value subtracted from every sample, in the data's unit.
#[gen_stub_pyclass]
#[pyclass(name = "SubtractBaseline", skip_from_py_object)]
#[derive(Clone)]
pub struct PySubtractBaseline {
    pub stage: PipelineStage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySubtractBaseline {
    /// See the class docs.
    #[new]
    fn new(baseline: f32) -> Self {
        Self { stage: PipelineStage::SubtractBaseline { baseline } }
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.stage)
    }
}

/// Pointwise clamp of `x` to `[min, max]`: a pipeline stage.
///
/// Parameters
/// ----------
/// min, max : float
///     Limits, in the data's unit.
#[gen_stub_pyclass]
#[pyclass(name = "Clamp", skip_from_py_object)]
#[derive(Clone)]
pub struct PyClamp {
    pub stage: PipelineStage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyClamp {
    /// See the class docs.
    #[new]
    fn new(min: f32, max: f32) -> Self {
        Self { stage: PipelineStage::Clamp { min, max } }
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.stage)
    }
}

/// `data · alpha + beta` of an array, on the device.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]` (or 1-D), converted to float32.
/// alpha : float
/// beta : float, default 0.0
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// numpy.ndarray
///     Float32, same shape as `data`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (data, alpha, beta=0.0, *, runtime=None))]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn scale_samples<'py>(py: Python<'py>, data: Bound<'py, PyAny>, alpha: f32, beta: f32, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PipelineStage::Scale { alpha, beta }, &data, None, runtime)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScale>()?;
    m.add_class::<PySubtractBaseline>()?;
    m.add_class::<PyClamp>()?;
    m.add_function(wrap_pyfunction!(scale_samples, m)?)?;
    Ok(())
}
