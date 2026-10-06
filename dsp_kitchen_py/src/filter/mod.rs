//! `dsp_kitchen.filter`: IIR (`iir`), FIR (`fir`), non-linear (`non_linear`) and template
//! (`template`) filters. Every filter is a pipeline stage object and a function of the same name
//! in lower case; defaults are dsp-base's, which follow scipy.

pub mod fir;
pub mod iir;
pub mod non_linear;
pub mod template;

use dsp_base::core::EdgeMode;
use dsp_base::filter::{FilterMode, FilterStart};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `direction=`: `"forward-backward"` (zero phase, `sosfiltfilt`) or `"forward"` (causal,
/// `sosfilt`).
pub(crate) fn parse_direction(direction: &str) -> PyResult<FilterMode> {
    match direction {
        "forward-backward" => Ok(FilterMode::ForwardBackward),
        "forward" => Ok(FilterMode::Forward),
        other => Err(PyValueError::new_err(format!("direction must be 'forward-backward' or 'forward', got '{other}'"))),
    }
}

pub(crate) fn direction_name(mode: FilterMode) -> &'static str {
    match mode {
        FilterMode::ForwardBackward => "forward-backward",
        FilterMode::Forward => "forward",
    }
}

/// `start=`: `"rest"` (as if zero before the first sample, `sosfilt` without `zi`) or
/// `"steady-state"` (from the first sample's steady state, `zi = sosfilt_zi · x[0]`).
pub(crate) fn parse_start(start: &str) -> PyResult<FilterStart> {
    match start {
        "rest" => Ok(FilterStart::Rest),
        "steady-state" => Ok(FilterStart::SteadyState),
        other => Err(PyValueError::new_err(format!("start must be 'rest' or 'steady-state', got '{other}'"))),
    }
}

/// `edge=`: what a stencil reads past the ends (scipy names: `"zeros"` = `constant` 0, `"odd"`,
/// `"reflect"`, `"nearest"`).
pub(crate) fn parse_edge(edge: &str) -> PyResult<EdgeMode> {
    match edge {
        "zeros" => Ok(EdgeMode::Zeros),
        "odd" => Ok(EdgeMode::Odd),
        "reflect" => Ok(EdgeMode::Reflect),
        "nearest" => Ok(EdgeMode::Nearest),
        other => Err(PyValueError::new_err(format!("edge must be 'zeros', 'odd', 'reflect' or 'nearest', got '{other}'"))),
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

/// Adds every filter class and function to the (flat) native module; `dsp_kitchen.filter.*`
/// arranges them.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    iir::register(m)?;
    fir::register(m)?;
    non_linear::register(m)?;
    template::register(m)
}
