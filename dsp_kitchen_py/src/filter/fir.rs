//! `dsp_kitchen.filter.fir`: Gaussian smoothing (`scipy.ndimage.gaussian_filter1d`).

use dsp_base::filter::GAUSSIAN_DEFAULT_EDGE;
use dsp_base::pipeline::PipelineStage;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use super::{edge_name, parse_edge};
use crate::pipeline::run_stage;

/// Zero-phase Gaussian smoothing along time (as `scipy.ndimage.gaussian_filter1d`): a pipeline stage.
///
/// Parameters
/// ----------
/// sigma_samples : float
///     Standard deviation of the Gaussian, **samples**; the kernel is cut at 4 σ.
/// edge : {"zeros", "odd", "reflect", "nearest"}, default "reflect"
///     Values assumed beyond the ends: zeros; odd (point-symmetric, `2·x₀ − x`); reflect (mirrored,
///     the edge sample repeated: `d c b a | a b c d`); nearest (the edge sample repeated).
#[gen_stub_pyclass]
#[pyclass(name = "GaussianSmooth", skip_from_py_object)]
#[derive(Clone)]
pub struct PyGaussianSmooth {
    pub stage: PipelineStage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGaussianSmooth {
    /// See the class docs.
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

/// Gaussian smoothing of an array along time (as `scipy.ndimage.gaussian_filter1d`).
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]` (or 1-D), converted to float32.
/// sigma_samples : float
///     Standard deviation, samples; the kernel is cut at 4 σ.
/// edge : {"zeros", "odd", "reflect", "nearest"}, default "reflect"
///     Values assumed beyond the ends: zeros; odd (point-symmetric, `2·x₀ − x`); reflect (mirrored,
///     the edge sample repeated: `d c b a | a b c d`); nearest (the edge sample repeated).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// numpy.ndarray
///     Float32, same shape as `data`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (data, sigma_samples, *, edge=None, runtime=None))]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn gaussian_smooth<'py>(py: Python<'py>, data: Bound<'py, PyAny>, sigma_samples: f32, edge: Option<&str>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyGaussianSmooth::new(sigma_samples, edge)?.stage, &data, None, runtime)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGaussianSmooth>()?;
    m.add_function(wrap_pyfunction!(gaussian_smooth, m)?)?;
    Ok(())
}
