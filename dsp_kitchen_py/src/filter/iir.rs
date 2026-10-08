//! `dsp_kitchen.filter.iir`: Butterworth and Chebyshev I filters (any order, per edge as in scipy)
//! and notch filters, forward-backward (zero phase, default) or forward.

use dsp_base::filter::design::DEFAULT_BUTTERWORTH_ORDER;
use dsp_base::filter::{FilterBand, FilterDesign, FilterSpec};
use dsp_base::pipeline::PipelineStage;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

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
    ($rust:ident, $py:literal, $doc:literal) => {
        #[doc = $doc]
        #[gen_stub_pyclass]
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

filter_class!(
    PyBandpassFilter,
    "BandpassFilter",
    "Butterworth band-pass filter: a pipeline stage (see `Pipeline`).

Parameters
----------
low_hz, high_hz : float
    Pass band, Hz.
order : int, default 5
    Butterworth order **per edge**, as scipy (`butter`): a band filter of order 5 has 10 poles.
direction : {\"forward-backward\", \"forward\"}, default \"forward-backward\"
    `\"forward-backward\"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
    `\"forward\"`: causal (as `sosfilt`).
start : {\"rest\", \"steady-state\"}, default \"rest\"
    `\"rest\"`: the input is taken as zero before the first sample (a DC level gives a step
    transient). `\"steady-state\"`: from the first sample's steady state (as `sosfilt_zi`).

Examples
--------
>>> stage = BandpassFilter(300.0, 5000.0)
>>> y = Pipeline([stage]).run(x, fs=30000.0)"
);
filter_class!(
    PyHighpassFilter,
    "HighpassFilter",
    "Butterworth high-pass filter: a pipeline stage (see `Pipeline`).

Parameters
----------
cutoff_hz : float
    Cutoff, Hz.
order : int, default 5
    Butterworth order **per edge**, as scipy (`butter`): a band filter of order 5 has 10 poles.
direction : {\"forward-backward\", \"forward\"}, default \"forward-backward\"
    `\"forward-backward\"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
    `\"forward\"`: causal (as `sosfilt`).
start : {\"rest\", \"steady-state\"}, default \"rest\"
    `\"rest\"`: the input is taken as zero before the first sample (a DC level gives a step
    transient). `\"steady-state\"`: from the first sample's steady state (as `sosfilt_zi`).

Examples
--------
>>> y = Pipeline([HighpassFilter(300.0, order=3)]).run(x, fs=30000.0)"
);
filter_class!(
    PyLowpassFilter,
    "LowpassFilter",
    "Butterworth low-pass filter: a pipeline stage (see `Pipeline`).

Parameters
----------
cutoff_hz : float
    Cutoff, Hz.
order : int, default 5
    Butterworth order **per edge**, as scipy (`butter`): a band filter of order 5 has 10 poles.
direction : {\"forward-backward\", \"forward\"}, default \"forward-backward\"
    `\"forward-backward\"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
    `\"forward\"`: causal (as `sosfilt`).
start : {\"rest\", \"steady-state\"}, default \"rest\"
    `\"rest\"`: the input is taken as zero before the first sample (a DC level gives a step
    transient). `\"steady-state\"`: from the first sample's steady state (as `sosfilt_zi`).

Examples
--------
>>> y = Pipeline([LowpassFilter(5000.0)]).run(x, fs=30000.0)"
);
filter_class!(
    PyBandstopFilter,
    "BandstopFilter",
    "Butterworth band-stop filter: a pipeline stage (see `Pipeline`).

Parameters
----------
low_hz, high_hz : float
    Stop band, Hz.
order : int, default 5
    Butterworth order **per edge**, as scipy (`butter`): a band filter of order 5 has 10 poles.
direction : {\"forward-backward\", \"forward\"}, default \"forward-backward\"
    `\"forward-backward\"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
    `\"forward\"`: causal (as `sosfilt`).
start : {\"rest\", \"steady-state\"}, default \"rest\"
    `\"rest\"`: the input is taken as zero before the first sample (a DC level gives a step
    transient). `\"steady-state\"`: from the first sample's steady state (as `sosfilt_zi`).

Examples
--------
>>> y = Pipeline([BandstopFilter(45.0, 55.0)]).run(x, fs=30000.0)"
);
filter_class!(
    PyNotchFilter,
    "NotchFilter",
    "Second-order notch filter (as `scipy.signal.iirnotch`): a pipeline stage (see `Pipeline`).

Parameters
----------
freq_hz : float
    Frequency removed, Hz.
q : float
    Quality factor: the notch width is `freq_hz / q`.
direction : {\"forward-backward\", \"forward\"}, default \"forward-backward\"
    `\"forward-backward\"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
    `\"forward\"`: causal (as `sosfilt`).
start : {\"rest\", \"steady-state\"}, default \"rest\"
    `\"rest\"`: the input is taken as zero before the first sample (a DC level gives a step
    transient). `\"steady-state\"`: from the first sample's steady state (as `sosfilt_zi`).

Examples
--------
>>> y = Pipeline([NotchFilter(60.0, 30.0)]).run(x, fs=30000.0)"
);
filter_class!(
    PyChebyshevFilter,
    "ChebyshevFilter",
    "Chebyshev type I filter (as `scipy.signal.cheby1`): a pipeline stage (see `Pipeline`).

Parameters
----------
order : int
    Filter order (per edge for band filters, as scipy).
ripple_db : float
    Pass-band ripple, dB.
btype : {\"lowpass\", \"highpass\", \"bandpass\", \"bandstop\"}
low_hz : float
    Cutoff, Hz (the lower edge for band filters).
high_hz : float, optional
    Upper edge, Hz (band filters only).
direction : {\"forward-backward\", \"forward\"}, default \"forward-backward\"
    `\"forward-backward\"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
    `\"forward\"`: causal (as `sosfilt`).
start : {\"rest\", \"steady-state\"}, default \"rest\"
    `\"rest\"`: the input is taken as zero before the first sample (a DC level gives a step
    transient). `\"steady-state\"`: from the first sample's steady state (as `sosfilt_zi`).

Examples
--------
>>> y = Pipeline([ChebyshevFilter(4, 0.5, \"highpass\", 300.0)]).run(x, fs=30000.0)"
);

#[gen_stub_pymethods]
#[pymethods]
impl PyBandpassFilter {
    /// See the class docs.
    #[new]
    #[pyo3(signature = (low_hz, high_hz, *, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(low_hz: f64, high_hz: f64, order: usize, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Butterworth { order, band: FilterBand::Bandpass(low_hz, high_hz) }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("BandpassFilter", &self.spec)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHighpassFilter {
    /// See the class docs.
    #[new]
    #[pyo3(signature = (cutoff_hz, *, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(cutoff_hz: f64, order: usize, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Butterworth { order, band: FilterBand::Highpass(cutoff_hz) }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("HighpassFilter", &self.spec)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLowpassFilter {
    /// See the class docs.
    #[new]
    #[pyo3(signature = (cutoff_hz, *, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(cutoff_hz: f64, order: usize, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Butterworth { order, band: FilterBand::Lowpass(cutoff_hz) }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("LowpassFilter", &self.spec)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBandstopFilter {
    /// See the class docs.
    #[new]
    #[pyo3(signature = (low_hz, high_hz, *, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(low_hz: f64, high_hz: f64, order: usize, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Butterworth { order, band: FilterBand::Bandstop(low_hz, high_hz) }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("BandstopFilter", &self.spec)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNotchFilter {
    /// See the class docs.
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

#[gen_stub_pymethods]
#[pymethods]
impl PyChebyshevFilter {
    /// See the class docs.
    #[new]
    #[pyo3(signature = (order, ripple_db, btype, low_hz, high_hz=None, *, direction=DEFAULT_DIRECTION, start=DEFAULT_START))]
    fn new(order: usize, ripple_db: f64, btype: &str, low_hz: f64, high_hz: Option<f64>, direction: &str, start: &str) -> PyResult<Self> {
        Ok(Self { spec: spec(FilterDesign::Chebyshev1 { order, ripple_db, band: band(btype, low_hz, high_hz)? }, direction, start)? })
    }
    fn __repr__(&self) -> String {
        repr("ChebyshevFilter", &self.spec)
    }
}

/// Butterworth band-pass filter of an array (one call; for several stages use `Pipeline`).
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]` (or 1-D), converted to float32; filtered along samples.
/// low_hz, high_hz : float
///     Pass band, Hz.
/// fs : float
///     Sampling rate of `data`, Hz.
/// order : int, default 5
///     Butterworth order **per edge**, as scipy (`butter`): a band filter of order 5 has 10 poles.
/// direction : {"forward-backward", "forward"}, default "forward-backward"
///     `"forward-backward"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
///     `"forward"`: causal (as `sosfilt`).
/// start : {"rest", "steady-state"}, default "rest"
///     `"rest"`: the input is taken as zero before the first sample (a DC level gives a step
///     transient). `"steady-state"`: from the first sample's steady state (as `sosfilt_zi`).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// numpy.ndarray
///     The filtered data, float32, same shape as `data`.
///
/// Examples
/// --------
/// >>> y = bandpass_filter(x, 300.0, 5000.0, fs=30000.0)
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (data, low_hz, high_hz, *, fs, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START, runtime=None))]
#[allow(clippy::too_many_arguments)]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn bandpass_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, low_hz: f64, high_hz: f64, fs: f64, order: usize, direction: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyBandpassFilter::new(low_hz, high_hz, order, direction, start)?.stage(), &data, Some(fs), runtime)
}

/// Butterworth high-pass filter of an array.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]` (or 1-D), converted to float32; filtered along samples.
/// cutoff_hz : float
///     Cutoff, Hz.
/// fs : float
///     Sampling rate of `data`, Hz.
/// order : int, default 5
///     Butterworth order **per edge**, as scipy (`butter`): a band filter of order 5 has 10 poles.
/// direction : {"forward-backward", "forward"}, default "forward-backward"
///     `"forward-backward"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
///     `"forward"`: causal (as `sosfilt`).
/// start : {"rest", "steady-state"}, default "rest"
///     `"rest"`: the input is taken as zero before the first sample (a DC level gives a step
///     transient). `"steady-state"`: from the first sample's steady state (as `sosfilt_zi`).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// numpy.ndarray
///     The filtered data, float32, same shape as `data`.
///
/// Examples
/// --------
/// >>> y = highpass_filter(x, 300.0, fs=30000.0)
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (data, cutoff_hz, *, fs, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START, runtime=None))]
#[allow(clippy::too_many_arguments)]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn highpass_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, cutoff_hz: f64, fs: f64, order: usize, direction: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyHighpassFilter::new(cutoff_hz, order, direction, start)?.stage(), &data, Some(fs), runtime)
}

/// Butterworth low-pass filter of an array.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]` (or 1-D), converted to float32; filtered along samples.
/// cutoff_hz : float
///     Cutoff, Hz.
/// fs : float
///     Sampling rate of `data`, Hz.
/// order : int, default 5
///     Butterworth order **per edge**, as scipy (`butter`): a band filter of order 5 has 10 poles.
/// direction : {"forward-backward", "forward"}, default "forward-backward"
///     `"forward-backward"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
///     `"forward"`: causal (as `sosfilt`).
/// start : {"rest", "steady-state"}, default "rest"
///     `"rest"`: the input is taken as zero before the first sample (a DC level gives a step
///     transient). `"steady-state"`: from the first sample's steady state (as `sosfilt_zi`).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// numpy.ndarray
///     The filtered data, float32, same shape as `data`.
///
/// Examples
/// --------
/// >>> y = lowpass_filter(x, 5000.0, fs=30000.0)
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (data, cutoff_hz, *, fs, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START, runtime=None))]
#[allow(clippy::too_many_arguments)]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn lowpass_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, cutoff_hz: f64, fs: f64, order: usize, direction: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyLowpassFilter::new(cutoff_hz, order, direction, start)?.stage(), &data, Some(fs), runtime)
}

/// Butterworth band-stop filter of an array.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]` (or 1-D), converted to float32; filtered along samples.
/// low_hz, high_hz : float
///     Stop band, Hz.
/// fs : float
///     Sampling rate of `data`, Hz.
/// order : int, default 5
///     Butterworth order **per edge**, as scipy (`butter`): a band filter of order 5 has 10 poles.
/// direction : {"forward-backward", "forward"}, default "forward-backward"
///     `"forward-backward"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
///     `"forward"`: causal (as `sosfilt`).
/// start : {"rest", "steady-state"}, default "rest"
///     `"rest"`: the input is taken as zero before the first sample (a DC level gives a step
///     transient). `"steady-state"`: from the first sample's steady state (as `sosfilt_zi`).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// numpy.ndarray
///     The filtered data, float32, same shape as `data`.
///
/// Examples
/// --------
/// >>> y = bandstop_filter(x, 45.0, 55.0, fs=30000.0)
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (data, low_hz, high_hz, *, fs, order=DEFAULT_BUTTERWORTH_ORDER, direction=DEFAULT_DIRECTION, start=DEFAULT_START, runtime=None))]
#[allow(clippy::too_many_arguments)]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
fn bandstop_filter<'py>(py: Python<'py>, data: Bound<'py, PyAny>, low_hz: f64, high_hz: f64, fs: f64, order: usize, direction: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    run_stage(py, PyBandstopFilter::new(low_hz, high_hz, order, direction, start)?.stage(), &data, Some(fs), runtime)
}

/// Second-order notch filter of an array (as `scipy.signal.iirnotch`).
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]` (or 1-D), converted to float32; filtered along samples.
/// freq_hz : float
///     Frequency removed, Hz.
/// q : float
///     Quality factor: the notch width is `freq_hz / q`.
/// fs : float
///     Sampling rate of `data`, Hz.
/// direction : {"forward-backward", "forward"}, default "forward-backward"
///     `"forward-backward"`: zero phase (as `scipy.signal.sosfiltfilt`; the effective order doubles).
///     `"forward"`: causal (as `sosfilt`).
/// start : {"rest", "steady-state"}, default "rest"
///     `"rest"`: the input is taken as zero before the first sample (a DC level gives a step
///     transient). `"steady-state"`: from the first sample's steady state (as `sosfilt_zi`).
/// runtime : str, optional
///     Compute runtime (`"wgpu"`, `"cuda"`, `"cpu"`, …); default: the current one.
///
/// Returns
/// -------
/// numpy.ndarray
///     The filtered data, float32, same shape as `data`.
///
/// Examples
/// --------
/// >>> y = notch_filter(x, 60.0, 30.0, fs=30000.0)
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (data, freq_hz, q, *, fs, direction=DEFAULT_DIRECTION, start=DEFAULT_START, runtime=None))]
#[allow(clippy::too_many_arguments)]
#[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float32]", imports = ("numpy", "numpy.typing")))]
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
