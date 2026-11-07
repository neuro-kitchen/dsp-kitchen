//! Python bindings of dspk-synapse: `dspk._native.synapse` and its submodules, arranged by the
//! pure-Python package `dspk.synapse`.

pub mod detection;
pub mod features;
pub mod io;
pub mod metrics;
pub mod sorting;

use pyo3::prelude::*;

/// Adds `synapse` (with `io`, `detection`, `extraction`, `features`, `metrics`, `clustering`,
/// `matching`, `spatial`) to the extension
/// module `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let synapse = PyModule::new(m.py(), "synapse")?;
    for (name, register) in [
        ("io", io::register as fn(&Bound<'_, PyModule>) -> PyResult<()>),
        ("detection", detection::register),
        ("extraction", features::register_extraction),
        ("features", features::register_features),
        ("metrics", metrics::register),
        ("clustering", sorting::register_clustering),
        ("matching", sorting::register_matching),
        ("spatial", sorting::register_spatial),
    ] {
        let sub = PyModule::new(m.py(), name)?;
        register(&sub)?;
        synapse.add_submodule(&sub)?;
    }
    m.add_submodule(&synapse)?;
    Ok(())
}
