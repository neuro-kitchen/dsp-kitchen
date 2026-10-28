//! Python bindings of the dspk library crates (dspk-core, dspk-io, dspk-ops). Each module adds its
//! classes and functions to a submodule of the `dspk._native` extension through [`register`]; the
//! pure-Python package `dspk` arranges them.

pub mod error;
pub mod io;
pub mod runtime;

use pyo3::prelude::*;

/// Adds the `runtime` and `io` submodules (and, later, `ops`) to the extension module `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let runtime = PyModule::new(m.py(), "runtime")?;
    runtime::register(&runtime)?;
    m.add_submodule(&runtime)?;
    let io = PyModule::new(m.py(), "io")?;
    io::register(&io)?;
    m.add_submodule(&io)?;
    Ok(())
}
