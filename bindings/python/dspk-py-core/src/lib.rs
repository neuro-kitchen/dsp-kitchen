//! Python bindings of the dspk library crates (dspk-core, dspk-io, dspk-ops, dspk-pipeline). Each module adds its
//! classes and functions to a submodule of the `dspk._native` extension through [`register`]; the
//! pure-Python package `dspk` arranges them.

pub mod error;
pub mod io;
pub mod ops;
pub mod pipeline;
pub mod runtime;

use pyo3::prelude::*;

/// Adds the `runtime`, `io`, `ops` and `pipeline` submodules to the extension module `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let runtime = PyModule::new(m.py(), "runtime")?;
    runtime::register(&runtime)?;
    m.add_submodule(&runtime)?;
    let io = PyModule::new(m.py(), "io")?;
    io::register(&io)?;
    m.add_submodule(&io)?;
    let ops = PyModule::new(m.py(), "ops")?;
    ops::register(&ops)?;
    m.add_submodule(&ops)?;
    let pipeline = PyModule::new(m.py(), "pipeline")?;
    pipeline::register(&pipeline)?;
    m.add_submodule(&pipeline)?;
    Ok(())
}
