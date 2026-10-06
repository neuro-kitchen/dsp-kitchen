//! Compute runtime selection for Python (`dsp_kitchen.runtime`: `available`, `current`, `set`).
//!
//! One rule for every native call: an explicit `runtime=` argument, else the runtime set with
//! `runtime.set(...)`, else `DSP_KITCHEN_RUNTIME`, else the first compiled-in runtime (GPUs
//! first).

use std::sync::Mutex;

use dsp_core::ComputeTarget;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The runtime chosen with `runtime.set`, if any.
static SELECTED: Mutex<Option<ComputeTarget>> = Mutex::new(None);

fn parse(name: &str) -> PyResult<ComputeTarget> {
    ComputeTarget::parse(name).and_then(ComputeTarget::checked).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The runtime a call uses: `explicit` if given, else the selected one, else the default.
pub fn target(explicit: Option<&str>) -> PyResult<ComputeTarget> {
    if let Some(name) = explicit {
        return parse(name);
    }
    if let Some(selected) = *SELECTED.lock().expect("runtime selection lock") {
        return Ok(selected);
    }
    ComputeTarget::from_env().map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Names of the runtimes compiled into this build, in preference order (GPUs first).
#[pyfunction]
pub fn available_runtimes() -> Vec<&'static str> {
    ComputeTarget::available().into_iter().map(ComputeTarget::name).collect()
}

/// The runtime calls use when they are not given `runtime=`.
#[pyfunction]
pub fn current_runtime() -> PyResult<&'static str> {
    Ok(target(None)?.name())
}

/// Uses `name` (`"wgpu"`, `"cpu"`, `"cuda"`, `"hip"`) for every later call; `None` returns to
/// the default (`DSP_KITCHEN_RUNTIME`, else the first compiled in).
#[pyfunction]
#[pyo3(signature = (name))]
pub fn set_runtime(name: Option<&str>) -> PyResult<()> {
    let chosen = name.map(parse).transpose()?;
    *SELECTED.lock().expect("runtime selection lock") = chosen;
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(available_runtimes, m)?)?;
    m.add_function(wrap_pyfunction!(current_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(set_runtime, m)?)?;
    Ok(())
}
