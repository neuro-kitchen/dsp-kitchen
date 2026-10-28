//! `dspk.ops.filter`: IIR filters as second-order sections (scipy semantics), on the device.
//!
//! `mode="forward_backward"` (default) is zero phase, like `scipy.signal.sosfiltfilt`: both edges
//! odd-padded by `settling_samples(sos)` (at most `samples - 1`), each pass started from the steady
//! state of its first sample. `mode="forward"` is causal, like `sosfilt`, started at rest (`zi`
//! absent) or, with `start="steady_state"`, from `sosfilt_zi · x[0]`.

use dspk_ops::core::buffer;
use dspk_ops::filter::design::{DEFAULT_BUTTERWORTH_ORDER, DEFAULT_SETTLING_TOLERANCE};
use dspk_ops::filter::{execute_filter, FilterBand, FilterDesign, FilterError, FilterMode, FilterSpec, FilterStart, Section, Sos};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::runtime;

fn filter_err(e: FilterError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

pub(super) fn band(kind: &str, cutoff: &[f64]) -> PyResult<FilterBand> {
    match (kind, cutoff) {
        ("lowpass", [f]) => Ok(FilterBand::Lowpass(*f)),
        ("highpass", [f]) => Ok(FilterBand::Highpass(*f)),
        ("bandpass", [lo, hi]) => Ok(FilterBand::Bandpass(*lo, *hi)),
        ("bandstop", [lo, hi]) => Ok(FilterBand::Bandstop(*lo, *hi)),
        ("lowpass" | "highpass", _) => Err(PyValueError::new_err(format!("{kind} takes one cutoff frequency, got {}", cutoff.len()))),
        ("bandpass" | "bandstop", _) => Err(PyValueError::new_err(format!("{kind} takes two cutoff frequencies (low, high), got {}", cutoff.len()))),
        _ => Err(PyValueError::new_err(format!("unknown band {kind:?} (lowpass, highpass, bandpass, bandstop)"))),
    }
}

pub(super) fn design(name: &str, order: usize, band: FilterBand, ripple_db: Option<f64>) -> PyResult<FilterDesign> {
    match (name, ripple_db) {
        ("butterworth", None) => Ok(FilterDesign::Butterworth { order, band }),
        ("bessel", None) => Ok(FilterDesign::Bessel { order, band }),
        ("chebyshev1", Some(ripple_db)) => Ok(FilterDesign::Chebyshev1 { order, ripple_db, band }),
        ("chebyshev1", None) => Err(PyValueError::new_err("design=\"chebyshev1\" needs ripple_db")),
        ("butterworth" | "bessel", Some(_)) => Err(PyValueError::new_err(format!("ripple_db applies to chebyshev1 only, not {name}"))),
        _ => Err(PyValueError::new_err(format!("unknown design {name:?} (butterworth, chebyshev1, bessel)"))),
    }
}

pub(super) fn mode(name: &str) -> PyResult<FilterMode> {
    match name {
        "forward_backward" => Ok(FilterMode::ForwardBackward),
        "forward" => Ok(FilterMode::Forward),
        other => Err(PyValueError::new_err(format!("unknown mode {other:?} (forward_backward, forward)"))),
    }
}

pub(super) fn start(name: &str) -> PyResult<FilterStart> {
    match name {
        "rest" => Ok(FilterStart::Rest),
        "steady_state" => Ok(FilterStart::SteadyState),
        other => Err(PyValueError::new_err(format!("unknown start {other:?} (rest, steady_state)"))),
    }
}

fn sos_to_numpy<'py>(py: Python<'py>, sos: &Sos) -> Bound<'py, PyArray2<f64>> {
    let rows: Vec<f64> = sos.sections.iter().flat_map(|s| s.b.into_iter().chain(s.a)).collect();
    Array2::from_shape_vec((sos.len(), 6), rows).expect("6 coefficients per section").into_pyarray(py)
}

pub(super) fn sos_from_numpy(sos: PyReadonlyArray2<'_, f64>) -> PyResult<Sos> {
    let view = sos.as_array();
    if view.ncols() != 6 {
        return Err(PyValueError::new_err(format!("sos must have shape (sections, 6), got {:?}", view.dim())));
    }
    let sections = view.rows().into_iter().map(|r| Section { b: [r[0], r[1], r[2]], a: [r[3], r[4], r[5]] }).collect();
    let sos = Sos::new(sections);
    sos.validate().map_err(filter_err)?;
    Ok(sos)
}

/// Filters `data` `(channels, samples)` with `spec` on `runtime` (see `dspk.runtime`).
fn run<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, spec: FilterSpec, sample_rate: f64, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let target = runtime::target(runtime)?;
    spec.design(sample_rate).map_err(filter_err)?;
    let (channels, samples) = data.as_array().dim();
    let values: Vec<f32> = data.as_array().iter().copied().collect();
    let out = py
        .detach(move || -> Result<Vec<f32>, String> {
            let client = runtime::open_client(target)?;
            let input = buffer::upload(&client, &values);
            let output = buffer::empty::<f32>(&client, values.len());
            execute_filter::<f32>(&client, &spec, sample_rate, &input, &output, channels, samples).map_err(|e| e.to_string())?;
            Ok(buffer::download::<f32>(&client, output))
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(Array2::from_shape_vec((channels, samples), out).expect("one value per input value").into_pyarray(py))
}

/// Second-order sections `(sections, 6)` `[b0, b1, b2, a0, a1, a2]` (scipy layout) of an IIR design.
///
/// `band`: lowpass, highpass (one `cutoff` in Hz), bandpass, bandstop (`(low, high)` in Hz).
/// `design`: butterworth, chebyshev1 (needs `ripple_db`), bessel.
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
#[pyo3(signature = (band, cutoff, sample_rate, *, order=DEFAULT_BUTTERWORTH_ORDER, design="butterworth", ripple_db=None))]
pub fn sos<'py>(py: Python<'py>, band: &str, cutoff: Vec<f64>, sample_rate: f64, order: usize, design: &str, ripple_db: Option<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let spec = FilterSpec { design: self::design(design, order, self::band(band, &cutoff)?, ripple_db)?, mode: FilterMode::default(), start: FilterStart::default() };
    Ok(sos_to_numpy(py, &spec.design(sample_rate).map_err(filter_err)?))
}

/// Designs an IIR filter and applies it to `data` `(channels, samples)` on the device.
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
#[pyo3(signature = (data, sample_rate, band, cutoff, *, order=DEFAULT_BUTTERWORTH_ORDER, design="butterworth", ripple_db=None, mode="forward_backward", start="rest", runtime=None))]
#[allow(clippy::too_many_arguments)]
pub fn iir<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'_, f32>,
    sample_rate: f64,
    band: &str,
    cutoff: Vec<f64>,
    order: usize,
    design: &str,
    ripple_db: Option<f64>,
    mode: &str,
    start: &str,
    runtime: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let spec = FilterSpec { design: self::design(design, order, self::band(band, &cutoff)?, ripple_db)?, mode: self::mode(mode)?, start: self::start(start)? };
    run(py, data, spec, sample_rate, runtime)
}

/// Second-order notch at `freq` Hz with quality factor `q` (`scipy.signal.iirnotch`), applied on
/// the device.
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
#[pyo3(signature = (data, sample_rate, freq, q, *, mode="forward_backward", start="rest", runtime=None))]
#[allow(clippy::too_many_arguments)]
pub fn notch<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, sample_rate: f64, freq: f64, q: f64, mode: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let spec = FilterSpec { design: FilterDesign::Notch { freq_hz: freq, q }, mode: self::mode(mode)?, start: self::start(start)? };
    run(py, data, spec, sample_rate, runtime)
}

/// Applies second-order sections `sos` `(sections, 6)` (scipy layout) to `data` on the device.
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
#[pyo3(signature = (data, sos, *, mode="forward_backward", start="rest", runtime=None))]
pub fn sosfilt<'py>(py: Python<'py>, data: PyReadonlyArray2<'_, f32>, sos: PyReadonlyArray2<'_, f64>, mode: &str, start: &str, runtime: Option<&str>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let spec = FilterSpec { design: FilterDesign::Sos(sos_from_numpy(sos)?), mode: self::mode(mode)?, start: self::start(start)? };
    // Explicit sections are already designed for their sample rate; any positive rate passes
    run(py, data, spec, 1.0, runtime)
}

/// Samples after which the impulse response of `sos` has decayed below the settling tolerance:
/// the odd padding of each edge in `mode="forward_backward"` (at most `samples - 1`).
#[gen_stub_pyfunction(module = "dspk._native.ops.filter")]
#[pyfunction]
pub fn settling_samples(sos: PyReadonlyArray2<'_, f64>) -> PyResult<usize> {
    Ok(sos_from_numpy(sos)?.settling_samples(DEFAULT_SETTLING_TOLERANCE))
}

pyo3_stub_gen::module_variable!("dspk._native.ops.filter", "DEFAULT_ORDER", usize);
pyo3_stub_gen::module_variable!("dspk._native.ops.filter", "SETTLING_TOLERANCE", f64);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sos, m)?)?;
    m.add_function(wrap_pyfunction!(iir, m)?)?;
    m.add_function(wrap_pyfunction!(notch, m)?)?;
    m.add_function(wrap_pyfunction!(sosfilt, m)?)?;
    m.add_function(wrap_pyfunction!(settling_samples, m)?)?;
    m.add("DEFAULT_ORDER", DEFAULT_BUTTERWORTH_ORDER)?;
    m.add("SETTLING_TOLERANCE", DEFAULT_SETTLING_TOLERANCE)?;
    Ok(())
}
