//! `dsp_kitchen.filter.non_linear`: running median (`scipy.signal.medfilt`) and the
//! Teager-Kaiser energy operator.

use dsp_base::filter::non_linear::{MEDIAN9_RADIUS, MEDIAN_DEFAULT_EDGE, TEAGER_KAISER_DEFAULT_EDGE};
use dsp_base::pipeline::PipelineStage;
use pyo3::prelude::*;

use super::{edge_name, parse_edge};
use crate::pipeline::run_stage;

/// Default median width: 9 points (the branch-free network).
const DEFAULT_MEDIAN_WIDTH: usize = 2 * MEDIAN9_RADIUS + 1;

/// Running median over an odd `width` along time; `edge` defaults to `"zeros"` as in
/// `scipy.signal.medfilt`.
#[pyclass(name = "MedianFilter", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMedianFilter {
    pub stage: PipelineStage,
}

#[pymethods]
impl PyMedianFilter {
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

/// Teager-Kaiser energy `x[t]² − x[t−1]·x[t+1]`; `edge` (the ends' missing neighbours)
/// defaults to `"reflect"`.
#[pyclass(name = "TeagerKaiser", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTeagerKaiser {
    pub stage: PipelineStage,
}

#[pymethods]
impl PyTeagerKaiser {
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

/// Running median of `data` (`[channels, samples]`, or 1-D) along time.
#[pyfunction]
#[pyo3(signature = (data, width=DEFAULT_MEDIAN_WIDTH, *, edge=None, runtime=None))]
fn median_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, width: usize, edge: Option<&str>, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyMedianFilter::new(width, edge)?.stage, &data, None, runtime)
}

/// Teager-Kaiser energy of `data` along time.
#[pyfunction]
#[pyo3(signature = (data, *, edge=None, runtime=None))]
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
