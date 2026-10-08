//! `dsp_kitchen.filter.non_linear`: running median (`scipy.signal.medfilt`) and the
//! Teager-Kaiser energy operator.

use dsp_base::filter::non_linear::{MEDIAN9_RADIUS, MEDIAN_DEFAULT_EDGE, TEAGER_KAISER_DEFAULT_EDGE};
use dsp_base::pipeline::PipelineStage;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use super::{edge_name, parse_edge};
use crate::pipeline::run_stage;

/// Default median width: 9 points (the branch-free network).
const DEFAULT_MEDIAN_WIDTH: usize = 2 * MEDIAN9_RADIUS + 1;

/// Running median along time (as `scipy.signal.medfilt`): a pipeline stage. Removes spikes and
/// artefacts narrower than half the window while keeping edges.
///
/// Parameters
/// ----------
/// width : int, default 9
///     Window, samples (odd, at most 31; 9 uses a branch-free sorting network).
/// edge : {"zeros", "odd", "reflect", "nearest"}, default "zeros"
///     Values assumed beyond the ends: zeros; odd (point-symmetric, `2·x₀ − x`); reflect (mirrored,
///     the edge sample repeated: `d c b a | a b c d`); nearest (the edge sample repeated).
#[gen_stub_pyclass]
#[pyclass(name = "MedianFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMedianFilter {
    pub stage: PipelineStage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMedianFilter {
    /// See the class docs.
    #[new]
    #[pyo3(signature = (width=DEFAULT_MEDIAN_WIDTH, *, edge=None))]
    fn new(width: usize, edge: Option<&str>) -> PyResult<Self> {
        let edge = edge.map(parse_edge).transpose()?.unwrap_or(MEDIAN_DEFAULT_EDGE);
        Ok(Self { stage: PipelineStage::Median { width, edge } })
    }
    fn __repr__(&self) -> String {
        match &self.stage {
            PipelineStage::Median { width, edge } => format!("MedianFilter(width={width}, edge='{}')", edge_name(*edge)),
            _ => unreachable!("a Median stage"),
        }
    }
}

/// Teager-Kaiser energy operator `x[t]² − x[t−1]·x[t+1]`: a pipeline stage. Large for short,
/// high-frequency bursts (spikes, MUAPs); used to sharpen detection.
///
/// Parameters
/// ----------
/// edge : {"zeros", "odd", "reflect", "nearest"}, default "reflect"
///     The missing neighbours at the ends: zeros; odd (point-symmetric, `2·x₀ − x`); reflect (mirrored,
///     the edge sample repeated: `d c b a | a b c d`); nearest (the edge sample repeated).
#[gen_stub_pyclass]
#[pyclass(name = "TeagerKaiser", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTeagerKaiser {
    pub stage: PipelineStage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTeagerKaiser {
    /// See the class docs.
    #[new]
    #[pyo3(signature = (*, edge=None))]
    fn new(edge: Option<&str>) -> PyResult<Self> {
        let edge = edge.map(parse_edge).transpose()?.unwrap_or(TEAGER_KAISER_DEFAULT_EDGE);
        Ok(Self { stage: PipelineStage::TeagerKaiser { edge } })
    }
    fn __repr__(&self) -> String {
        match &self.stage {
            PipelineStage::TeagerKaiser { edge } => format!("TeagerKaiser(edge='{}')", edge_name(*edge)),
            _ => unreachable!("a TeagerKaiser stage"),
        }
    }
}

/// Running median of an array along time (as `scipy.signal.medfilt`).
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]` (or 1-D), converted to float32.
/// width : int, default 9
///     Window, samples (odd, at most 31).
/// edge : {"zeros", "odd", "reflect", "nearest"}, default "zeros"
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
#[pyo3(signature = (data, width=DEFAULT_MEDIAN_WIDTH, *, edge=None, runtime=None))]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn median_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, width: usize, edge: Option<&str>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyMedianFilter::new(width, edge)?.stage, &data, None, runtime)
}

/// Teager-Kaiser energy of an array along time, `x[t]² − x[t−1]·x[t+1]`.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]` (or 1-D), converted to float32.
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
#[pyo3(signature = (data, *, edge=None, runtime=None))]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn teager_kaiser_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, edge: Option<&str>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyTeagerKaiser::new(edge)?.stage, &data, None, runtime)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMedianFilter>()?;
    m.add_class::<PyTeagerKaiser>()?;
    m.add_function(wrap_pyfunction!(median_filter, m)?)?;
    m.add_function(wrap_pyfunction!(teager_kaiser_filter, m)?)?;
    Ok(())
}
