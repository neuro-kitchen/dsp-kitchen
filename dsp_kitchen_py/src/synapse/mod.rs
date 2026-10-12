//! `dsp_kitchen.synapse`: probes, detection, extraction, localization and drift, clustering and
//! matching, metrics, sorting files, streaming detection.

pub mod comparison;
pub mod detection;
pub mod extraction;
pub mod metrics;
pub mod ml;
pub mod mountainsort5;
pub mod spykingcircus2;
pub mod tridesclous2;
pub mod probe;
pub mod sorting;
pub mod spatial;
pub mod storage;
pub mod streaming;

pub use probe::PyProbeLayout;

/// Adds every synapse class and function to the (flat) native module.
pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    probe::register(m)?;
    detection::register(m)?;
    extraction::register(m)?;
    spatial::register(m)?;
    sorting::register(m)?;
    metrics::register(m)?;
    comparison::register(m)?;
    storage::register(m)?;
    streaming::register(m)?;
    ml::register(m)?;
    mountainsort5::register(m)?;
    spykingcircus2::register(m)?;
    tridesclous2::register(m)
}
