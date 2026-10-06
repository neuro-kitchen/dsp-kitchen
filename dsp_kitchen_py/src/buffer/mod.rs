//! `dsp_kitchen.io`: recordings (any dsp-io format, lazily sliced) and memory-mapped raw files
//! with zero-copy NumPy views.

pub mod mmap;
pub mod recording;
pub mod synthetic;

pub use mmap::PyMmapRecording;
pub use recording::PyRecording;

pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    mmap::register(m)?;
    recording::register(m)?;
    synthetic::register(m)
}
