//! Python bindings of dspk-synapse: `dspk._native.synapse` and its submodules, arranged by the
//! pure-Python package `dspk.synapse`.

pub mod io;
pub mod metrics;

use pyo3::prelude::*;

/// Adds `synapse` (with `io`, `metrics`) to the extension module `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let synapse = PyModule::new(m.py(), "synapse")?;
    for (name, register) in [("io", io::register as fn(&Bound<'_, PyModule>) -> PyResult<()>), ("metrics", metrics::register)] {
        let sub = PyModule::new(m.py(), name)?;
        register(&sub)?;
        synapse.add_submodule(&sub)?;
    }
    m.add_submodule(&synapse)?;
    Ok(())
}
