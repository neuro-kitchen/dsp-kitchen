//! `dsp_kitchen.filter.iir`: Butterworth and Chebyshev I filters (any order, per edge as in scipy)
//! and notch filters, forward-backward (zero phase, default) or forward.

use dsp_base::filter::design::DEFAULT_BUTTERWORTH_ORDER;
use dsp_base::filter::{FilterBand, FilterDesign, FilterSpec};
use dsp_base::pipeline::PipelineStage;
use pyo3::prelude::*;

use super::{direction_name, parse_direction, parse_start};
use crate::pipeline::run_stage;

/// The default `direction=` and `start=` (dsp-base `FilterMode` / `FilterStart` defaults).
const DEFAULT_DIRECTION: &str = "forward-backward";
const DEFAULT_START: &str = "rest";

fn spec(design: FilterDesign, direction: &str, start: &str) -> PyResult<FilterSpec> {
    let base = match design {
        FilterDesign::Butterworth { order, band } => FilterSpec::butterworth(order, band),
        FilterDesign::Chebyshev1 { order, ripple_db, band } => FilterSpec::chebyshev1(order, ripple_db, band),
        FilterDesign::Notch { freq_hz, q } => FilterSpec::notch(freq_hz, q),
        FilterDesign::Sos(sos) => FilterSpec::sos(sos),
    };
    Ok(base.with_mode(parse_direction(direction)?).with_start(parse_start(start)?))
}

fn repr(class: &str, s: &FilterSpec) -> String {
    let design = match &s.design {
        FilterDesign::Butterworth { order, band } => format!("{band:?}, order={order}"),
        FilterDesign::Chebyshev1 { order, ripple_db, band } => format!("{band:?}, order={order}, ripple_db={ripple_db}"),
        FilterDesign::Notch { freq_hz, q } => format!("freq_hz={freq_hz}, q={q}"),
        FilterDesign::Sos(sos) => format!("{} sections", sos.sections.len()),
    };
    format!("{class}({design}, direction='{}', start={:?})", direction_name(s.mode), s.start)
}

macro_rules! filter_class {
    ($rust:ident, $py:literal) => {
        #[pyclass(name = $py, skip_from_py_object)]
        #[derive(Clone)]
        pub struct $rust {
            pub spec: FilterSpec,
        }

        impl $rust {
            pub fn stage(&self) -> PipelineStage {
                PipelineStage::Filter(self.spec.clone())
            }
        }
    };
}

filter_class!(PyBandpassFilter, "BandpassFilter");
filter_class!(PyHighpassFilter, "HighpassFilter");
filter_class!(PyLowpassFilter, "LowpassFilter");
filter_class!(PyBandstopFilter, "BandstopFilter");
filter_class!(PyNotchFilter, "NotchFilter");
filter_class!(PyChebyshevFilter, "ChebyshevFilter");

#[pymethods]
impl PyBandpassFilter {
    /// Butterworth band-pass between `low_hz` and `high_hz`, `order` per edge.
    #[new]
    #[pyo3(signature = (low_hz, high_hz, *, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(low_hz: f64, high_hz: f64, order: usize, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Butterworth { order, band: FilterBand::Bandpass(low_hz, high_hz) }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("BandpassFilter", &self.spec)
    }
}

#[pymethods]
impl PyHighpassFilter {
    /// Butterworth high-pass above `cutoff_hz`.
    #[new]
    #[pyo3(signature = (cutoff_hz, *, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(cutoff_hz: f64, order: usize, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Butterworth { order, band: FilterBand::Highpass(cutoff_hz) }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("HighpassFilter", &self.spec)
    }
}

#[pymethods]
impl PyLowpassFilter {
    /// Butterworth low-pass below `cutoff_hz`.
    #[new]
    #[pyo3(signature = (cutoff_hz, *, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(cutoff_hz: f64, order: usize, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Butterworth { order, band: FilterBand::Lowpass(cutoff_hz) }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("LowpassFilter", &self.spec)
    }
}

#[pymethods]
impl PyBandstopFilter {
    /// Butterworth band-stop between `low_hz` and `high_hz`.
    #[new]
    #[pyo3(signature = (low_hz, high_hz, *, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(low_hz: f64, high_hz: f64, order: usize, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Butterworth { order, band: FilterBand::Bandstop(low_hz, high_hz) }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("BandstopFilter", &self.spec)
    }
}

#[pymethods]
impl PyNotchFilter {
    /// Second-order notch at `freq_hz` with quality factor `q` (`scipy.signal.iirnotch`).
    #[new]
    #[pyo3(signature = (freq_hz, q, *, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(freq_hz: f64, q: f64, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Notch { freq_hz, q }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("NotchFilter", &self.spec)
    }
}

fn band(kind: &str, low_hz: f64, high_hz: Option<f64>) -> PyResult<FilterBand> {
    let need_high = || high_hz.ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("btype '{kind}' needs high_hz")));
    Ok(match kind {
        "lowpass" => FilterBand::Lowpass(low_hz),
        "highpass" => FilterBand::Highpass(low_hz),
        "bandpass" => FilterBand::Bandpass(low_hz, need_high()?),
        "bandstop" => FilterBand::Bandstop(low_hz, need_high()?),
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!("btype must be lowpass, highpass, bandpass or bandstop, got '{other}'"))),
    })
}

#[pymethods]
impl PyChebyshevFilter {
    /// Chebyshev type I (`scipy.signal.cheby1`): `order`, `ripple_db` of pass-band ripple,
    /// `btype` with cutoff `low_hz` (and `high_hz` for band filters).
    #[new]
    #[pyo3(signature = (order, ripple_db, btype, low_hz, high_hz=None, *, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(order: usize, ripple_db: f64, btype: &str, low_hz: f64, high_hz: Option<f64>, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Chebyshev1 { order, ripple_db, band: band(btype, low_hz, high_hz)? }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("ChebyshevFilter", &self.spec)
    }
}

/// Butterworth band-pass of `data` (`[channels, samples]`, or 1-D) sampled at `fs` Hz.
#[pyfunction]
#[pyo3(signature = (data, low_hz, high_hz, *, fs, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START, runtime=None))]
#[allow(clippy::too_many_arguments)]
fn bandpass_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, low_hz: f64, high_hz: f64, fs: f64, order: usize, direction: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyBandpassFilter::new(low_hz, high_hz, order, direction, start)?.stage(), &data, Some(fs), runtime)
}

/// Butterworth high-pass of `data` sampled at `fs` Hz.
#[pyfunction]
#[pyo3(signature = (data, cutoff_hz, *, fs, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START, runtime=None))]
#[allow(clippy::too_many_arguments)]
fn highpass_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, cutoff_hz: f64, fs: f64, order: usize, direction: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyHighpassFilter::new(cutoff_hz, order, direction, start)?.stage(), &data, Some(fs), runtime)
}

/// Butterworth low-pass of `data` sampled at `fs` Hz.
#[pyfunction]
#[pyo3(signature = (data, cutoff_hz, *, fs, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START, runtime=None))]
#[allow(clippy::too_many_arguments)]
fn lowpass_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, cutoff_hz: f64, fs: f64, order: usize, direction: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyLowpassFilter::new(cutoff_hz, order, direction, start)?.stage(), &data, Some(fs), runtime)
}

/// Butterworth band-stop of `data` sampled at `fs` Hz.
#[pyfunction]
#[pyo3(signature = (data, low_hz, high_hz, *, fs, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START, runtime=None))]
#[allow(clippy::too_many_arguments)]
fn bandstop_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, low_hz: f64, high_hz: f64, fs: f64, order: usize, direction: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyBandstopFilter::new(low_hz, high_hz, order, direction, start)?.stage(), &data, Some(fs), runtime)
}

/// Notch of `data` at `freq_hz` with quality factor `q`, sampled at `fs` Hz.
#[pyfunction]
#[pyo3(signature = (data, freq_hz, q, *, fs, direction=DEFAULT_DIRECTION, start=DEFAULT_START, runtime=None))]
#[allow(clippy::too_many_arguments)]
fn notch_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, freq_hz: f64, q: f64, fs: f64, direction: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyNotchFilter::new(freq_hz, q, direction, start)?.stage(), &data, Some(fs), runtime)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBandpassFilter>()?;
    m.add_class::<PyHighpassFilter>()?;
    m.add_class::<PyLowpassFilter>()?;
    m.add_class::<PyBandstopFilter>()?;
    m.add_class::<PyNotchFilter>()?;
    m.add_class::<PyChebyshevFilter>()?;
    m.add_function(wrap_pyfunction!(bandpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(highpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(lowpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(bandstop_filter, m)?)?;
    m.add_function(wrap_pyfunction!(notch_filter, m)?)?;
    Ok(())
}
