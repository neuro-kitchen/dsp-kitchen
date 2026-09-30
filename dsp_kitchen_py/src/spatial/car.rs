use pyo3::prelude::*;
use crate::pipeline::PyPipeline;
use dsp_base::pipeline::{Pipeline, PipelineStage};

#[pyclass(name = "CommonAverageReference", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCommonAverageReference;

#[pymethods]
impl PyCommonAverageReference {
    #[new]
    pub fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "CommonAverageReference()".to_string()
    }
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn common_average_reference<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut pipe = Pipeline::new();
    pipe.add(PipelineStage::CommonAverageReference);
    let py_pipe = PyPipeline::from_stages(pipe.stages().to_vec());
    py_pipe.run(py, data, 30000.0, None)
}
