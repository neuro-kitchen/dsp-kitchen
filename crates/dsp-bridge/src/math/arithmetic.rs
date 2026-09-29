use pyo3::prelude::*;
use crate::pipeline::PyPipeline;
use dsp_base::pipeline::{Pipeline, PipelineStage};

#[pyclass(name = "Scale", skip_from_py_object)]
#[derive(Clone)]
pub struct PyScale {
    pub alpha: f32,
    pub beta: f32,
}

#[pymethods]
impl PyScale {
    #[new]
    #[pyo3(signature = (alpha=0.195, beta=0.0))]
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self { alpha, beta }
    }

    fn __repr__(&self) -> String {
        format!("Scale(alpha={}, beta={})", self.alpha, self.beta)
    }
}

#[pyclass(name = "SubtractBaseline", skip_from_py_object)]
#[derive(Clone)]
pub struct PySubtractBaseline {
    pub baseline_uv: f32,
}

#[pymethods]
impl PySubtractBaseline {
    #[new]
    #[pyo3(signature = (baseline_uv=0.0))]
    pub fn new(baseline_uv: f32) -> Self {
        Self { baseline_uv }
    }

    fn __repr__(&self) -> String {
        format!("SubtractBaseline(baseline_uv={})", self.baseline_uv)
    }
}

#[pyclass(name = "Clamp", skip_from_py_object)]
#[derive(Clone)]
pub struct PyClamp {
    pub min_val: f32,
    pub max_val: f32,
}

#[pymethods]
impl PyClamp {
    #[new]
    #[pyo3(signature = (min_val=-1000.0, max_val=1000.0))]
    pub fn new(min_val: f32, max_val: f32) -> Self {
        Self { min_val, max_val }
    }

    fn __repr__(&self) -> String {
        format!("Clamp(min_val={}, max_val={})", self.min_val, self.max_val)
    }
}

#[pyfunction]
#[pyo3(signature = (data, alpha=0.195, beta=0.0))]
pub fn scale_samples<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    alpha: f32,
    beta: f32,
) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::Scale { alpha, beta });
    let py_pipe = PyPipeline::from_stages(pipe.stages().to_vec());
    py_pipe.run(py, data, 30000.0, None)
}
