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
    // `dspk.io` opens the neural formats too once Synapse is built in
    dspk_py_core::io::use_formats(dspk_py_core::io::FormatSet {
        open: dspk_synapse::io::open,
        open_source: dspk_synapse::io::open_source,
        sources: dspk_synapse::io::sources,
        all: dspk_synapse::io::formats,
    });
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
