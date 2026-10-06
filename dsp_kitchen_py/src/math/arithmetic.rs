//! `dsp_kitchen.math`: pointwise stages (scale, baseline, clamp). No defaults: the values depend
//! on the recording.

use dsp_base::pipeline::PipelineStage;
use pyo3::prelude::*;

use crate::pipeline::run_stage;

/// `x · alpha + beta` (e.g. stored steps to physical units).
#[pyclass(name = "Scale", skip_from_py_object)]
#[derive(Clone)]
pub struct PyScale {
    pub stage: PipelineStage,
}

#[pymethods]
impl PyScale {
    #[new]
    #[pyo3(signature = (alpha, beta=0.0))]
    fn new(alpha: f32, beta: f32) -> Self {
        Self { stage: PipelineStage::Scale { alpha, beta } }
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.stage)
    }
}

/// `x − baseline`.
#[pyclass(name = "SubtractBaseline", skip_from_py_object)]
#[derive(Clone)]
pub struct PySubtractBaseline {
    pub stage: PipelineStage,
}

#[pymethods]
impl PySubtractBaseline {
    #[new]
    fn new(baseline: f32) -> Self {
        Self { stage: PipelineStage::SubtractBaseline { baseline } }
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.stage)
    }
}

/// `x` limited to `[min, max]`.
#[pyclass(name = "Clamp", skip_from_py_object)]
#[derive(Clone)]
pub struct PyClamp {
    pub stage: PipelineStage,
}

#[pymethods]
impl PyClamp {
    #[new]
    fn new(min: f32, max: f32) -> Self {
        Self { stage: PipelineStage::Clamp { min, max } }
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.stage)
    }
}

/// `data · alpha + beta` on the device.
#[pyfunction]
#[pyo3(signature = (data, alpha, beta=0.0, *, runtime=None))]
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
