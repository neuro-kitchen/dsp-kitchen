//! `dspk.ops.filter`: filters on the device, one file per family (mirroring `dspk_ops::filter`):
//! IIR (`iir.rs`), FIR and Gaussian (`fir.rs`), median and Teager–Kaiser (`non_linear.rs`),
//! template subtraction (`template.rs`). All register into the same Python submodule.
//!
//! The argument parsers here are shared with the pipeline stages (`super::pipeline`).

pub mod fir;
pub mod iir;
pub mod non_linear;
pub mod template;

use dspk_ops::core::EdgeMode;
use dspk_ops::filter::{FilterBand, FilterDesign, FilterError, FilterMode, FilterStart, Section, Sos};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn filter_err(e: FilterError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

pub(crate) fn band(kind: &str, cutoff: &[f64]) -> PyResult<FilterBand> {
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

pub(crate) fn design(name: &str, order: usize, band: FilterBand, ripple_db: Option<f64>) -> PyResult<FilterDesign> {
    match (name, ripple_db) {
        ("butterworth", None) => Ok(FilterDesign::Butterworth { order, band }),
        ("bessel", None) => Ok(FilterDesign::Bessel { order, band }),
        ("chebyshev1", Some(ripple_db)) => Ok(FilterDesign::Chebyshev1 { order, ripple_db, band }),
        ("chebyshev1", None) => Err(PyValueError::new_err("design=\"chebyshev1\" needs ripple_db")),
        ("butterworth" | "bessel", Some(_)) => Err(PyValueError::new_err(format!("ripple_db applies to chebyshev1 only, not {name}"))),
        _ => Err(PyValueError::new_err(format!("unknown design {name:?} (butterworth, chebyshev1, bessel)"))),
    }
}

pub(crate) fn mode(name: &str) -> PyResult<FilterMode> {
    match name {
        "forward_backward" => Ok(FilterMode::ForwardBackward),
        "forward" => Ok(FilterMode::Forward),
        other => Err(PyValueError::new_err(format!("unknown mode {other:?} (forward_backward, forward)"))),
    }
}

pub(crate) fn start(name: &str) -> PyResult<FilterStart> {
    match name {
        "rest" => Ok(FilterStart::Rest),
        "steady_state" => Ok(FilterStart::SteadyState),
        other => Err(PyValueError::new_err(format!("unknown start {other:?} (rest, steady_state)"))),
    }
}

pub(crate) fn edge(name: &str) -> PyResult<EdgeMode> {
    match name {
        "zeros" => Ok(EdgeMode::Zeros),
        "odd" => Ok(EdgeMode::Odd),
        "reflect" => Ok(EdgeMode::Reflect),
        "nearest" => Ok(EdgeMode::Nearest),
        other => Err(PyValueError::new_err(format!("unknown edge {other:?} (zeros, odd, reflect, nearest)"))),
    }
}

pub(crate) fn edge_name(edge: EdgeMode) -> &'static str {
    match edge {
        EdgeMode::Zeros => "zeros",
        EdgeMode::Odd => "odd",
        EdgeMode::Reflect => "reflect",
        EdgeMode::Nearest => "nearest",
    }
}

pub(crate) fn sos_from_numpy(sos: PyReadonlyArray2<'_, f64>) -> PyResult<Sos> {
    let view = sos.as_array();
    if view.ncols() != 6 {
        return Err(PyValueError::new_err(format!("sos must have shape (sections, 6), got {:?}", view.dim())));
    }
    let sections = view.rows().into_iter().map(|r| Section { b: [r[0], r[1], r[2]], a: [r[3], r[4], r[5]] }).collect();
    let sos = Sos::new(sections);
    sos.validate().map_err(filter_err)?;
    Ok(sos)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    iir::register(m)?;
    fir::register(m)?;
    non_linear::register(m)?;
    template::register(m)?;
    Ok(())
}
