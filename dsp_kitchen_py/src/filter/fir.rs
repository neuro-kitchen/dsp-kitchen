//! `dsp_kitchen.filter.fir`: Gaussian smoothing (`scipy.ndimage.gaussian_filter1d`).

use dsp_base::filter::GAUSSIAN_DEFAULT_EDGE;
use dsp_base::pipeline::PipelineStage;
use pyo3::prelude::*;

use super::{edge_name, parse_edge};
use crate::pipeline::run_stage;

/// Zero-phase Gaussian smoothing with standard deviation `sigma_samples` (samples), cut at
/// dsp-base's `GAUSSIAN_TRUNCATE` σ; `edge` defaults to `"reflect"` as in scipy.
#[pyclass(name = "GaussianSmooth", skip_from_py_object)]
#[derive(Clone)]
pub struct PyGaussianSmooth {
    pub stage: PipelineStage,
}

#[pymethods]
impl PyGaussianSmooth {
    #[new]
    #[pyo3(signature = (sigma_samples, *, edge=None))]
    fn new(sigma_samples: f32, edge: Option<&str>) -> PyResult<Self> {
        let edge = edge.map(parse_edge).transpose()?.unwrap_or(GAUSSIAN_DEFAULT_EDGE);
        Ok(Self { stage: PipelineStage::GaussianSmooth { sigma_samples, edge } })
    }
    fn __repr__(&self) -> String {
        match &self.stage {
            PipelineStage::GaussianSmooth { sigma_samples, edge } => format!("GaussianSmooth(sigma_samples={sigma_samples}, edge='{}')", edge_name(*edge)),
            _ => unreachable!("a GaussianSmooth stage"),
        }
    }
}

/// Gaussian smoothing of `data` (`[channels, samples]`, or 1-D) along time.
#[pyfunction]
#[pyo3(signature = (data, sigma_samples, *, edge=None, runtime=None))]
fn gaussian_smooth<'py>(py: Python<'py>, data: Bound<'py, PyAny>, sigma_samples: f32, edge: Option<&str>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyGaussianSmooth::new(sigma_samples, edge)?.stage, &data, None, runtime)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGaussianSmooth>()?;
    m.add_function(wrap_pyfunction!(gaussian_smooth, m)?)?;
    Ok(())
}
