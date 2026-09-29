pub mod buffer;
pub mod filter;
pub mod linalg;
pub mod math;
pub mod pipeline;
pub mod spatial;
pub mod synapse;

// Backward-compatible module alias
pub use buffer as mmap;

use pyo3::prelude::*;

use crate::buffer::PyMmapRecording;
use crate::filter::{
    bandpass_filter, median_filter_9p, notch_filter, teager_kaiser_filter, PyBandpassFilter,
    PyMedianFilter, PyNotchFilter, PyTeagerKaiser,
};
use crate::linalg::PyPca;
use crate::math::{scale_samples, PyClamp, PyScale, PySubtractBaseline};
use crate::pipeline::{PyDspSession, PyPipeline};
use crate::spatial::{common_average_reference, PyCommonAverageReference};
use crate::synapse::{detect_spikes, estimate_noise, PyProbeLayout, PySpikeEvent};

#[pymodule]
fn _dsp_kitchen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyProbeLayout>()?;
    m.add_class::<PySpikeEvent>()?;
    m.add_class::<PyMmapRecording>()?;
    m.add_class::<PyScale>()?;
    m.add_class::<PySubtractBaseline>()?;
    m.add_class::<PyClamp>()?;
    m.add_class::<PyNotchFilter>()?;
    m.add_class::<PyBandpassFilter>()?;
    m.add_class::<PyCommonAverageReference>()?;
    m.add_class::<PyMedianFilter>()?;
    m.add_class::<PyTeagerKaiser>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyDspSession>()?;
    m.add_class::<PyPca>()?;

    // Direct functions
    m.add_function(wrap_pyfunction!(notch_filter, m)?)?;
    m.add_function(wrap_pyfunction!(bandpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(common_average_reference, m)?)?;
    m.add_function(wrap_pyfunction!(scale_samples, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_9p, m)?)?;
    m.add_function(wrap_pyfunction!(teager_kaiser_filter, m)?)?;
    m.add_function(wrap_pyfunction!(detect_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_noise, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
