//! Compute runtime choice from Python (`dspk.runtime`: `available`, `current`, `set`).
//!
//! One rule for every native call: an explicit `runtime=` argument, else the runtime chosen with
//! `dspk.runtime.set(...)`, else `DSPK_RUNTIME`, else the first compiled-in runtime (GPUs first).

use std::sync::Mutex;

use dspk_core::ComputeTarget;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// The runtime chosen with `set`, if any.
static SELECTED: Mutex<Option<ComputeTarget>> = Mutex::new(None);

fn parse(name: &str) -> PyResult<ComputeTarget> {
    ComputeTarget::parse(name).and_then(ComputeTarget::checked).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The runtime a call uses: `explicit` if given, else the chosen one, else the default.
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
#[gen_stub_pyfunction(module = "dspk._native.runtime")]
#[pyfunction]
pub fn available() -> Vec<&'static str> {
    ComputeTarget::available().into_iter().map(ComputeTarget::name).collect()
}

/// The runtime calls use when they are not given `runtime=`.
#[gen_stub_pyfunction(module = "dspk._native.runtime")]
#[pyfunction]
pub fn current() -> PyResult<&'static str> {
    Ok(target(None)?.name())
}

/// Chooses the runtime of every later call that is not given `runtime=`.
///
/// `name`: `"wgpu"`, `"cpu"`, `"cuda"` or `"hip"` (one of `available()`); `None` goes back to the
/// default (`DSPK_RUNTIME`, else the first compiled in).
#[gen_stub_pyfunction(module = "dspk._native.runtime")]
#[pyfunction]
#[pyo3(signature = (name))]
pub fn set(name: Option<&str>) -> PyResult<()> {
    let chosen = name.map(parse).transpose()?;
    *SELECTED.lock().expect("runtime selection lock") = chosen;
    Ok(())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(available, m)?)?;
    m.add_function(wrap_pyfunction!(current, m)?)?;
    m.add_function(wrap_pyfunction!(set, m)?)?;
    Ok(())
}
