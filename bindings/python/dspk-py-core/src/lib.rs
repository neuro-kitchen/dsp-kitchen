//! Python bindings of the dspk library crates (dspk-core, dspk-io, dspk-ops). Each module adds its
//! classes and functions to a submodule of the `dspk._native` extension through [`register`]; the
//! pure-Python package `dspk` arranges them.

pub mod runtime;

use pyo3::prelude::*;

/// Adds the `runtime` submodule (and, later, `io` and `ops`) to the extension module `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let runtime = PyModule::new(m.py(), "runtime")?;
    runtime::register(&runtime)?;
    m.add_submodule(&runtime)?;
    Ok(())
}
