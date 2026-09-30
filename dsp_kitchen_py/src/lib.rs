pub mod buffer;
pub mod filter;
pub mod linalg;
pub mod math;
pub mod pipeline;
pub mod spatial;
pub mod synapse;
pub mod synapse_ml;

// Backward-compatible module alias
pub use buffer as mmap;

use pyo3::prelude::*;

use crate::buffer::{list_nwb_series, PyMmapRecording, PyNwbZarrRecording};
use crate::filter::{
    bandpass_filter, median_filter_9p, notch_filter, subtract_template, teager_kaiser_filter,
    PyBandpassFilter, PyMedianFilter, PyNotchFilter, PyTeagerKaiser, PyTemplateFilter,
};
use crate::linalg::PyPca;
use crate::math::{scale_samples, PyClamp, PyScale, PySubtractBaseline};
use crate::pipeline::{PyDspSession, PyPipeline};
use crate::spatial::{common_average_reference, PyCommonAverageReference};
use crate::synapse::{
    compute_isi, compute_snr, compute_template, deduplicate_spikes, detect_spikes,
    estimate_noise, extract_snippets, sort_recording, PyDeduplicatedSpike, PyProbeLayout,
    PySpikeEvent, PyStreamingSortResult, PyWaveformSnippet,
};
use crate::synapse_ml::{
    PyContrastiveWaveformEmbedder, PyDartsortVaeEmbedder, PyOnnxModelRunner,
    PySingleChannelDenoiser, PySpatiotemporalUnetDenoiser, PyUnitQualityClassifier,
};

fn register_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core & Classical Classes
    m.add_class::<PyProbeLayout>()?;
    m.add_class::<PySpikeEvent>()?;
    m.add_class::<PyDeduplicatedSpike>()?;
    m.add_class::<PyWaveformSnippet>()?;
    m.add_class::<PyStreamingSortResult>()?;
    m.add_class::<PyMmapRecording>()?;
    m.add_class::<PyNwbZarrRecording>()?;
    m.add_class::<PyScale>()?;
    m.add_class::<PySubtractBaseline>()?;
    m.add_class::<PyClamp>()?;
    m.add_class::<PyNotchFilter>()?;
    m.add_class::<PyBandpassFilter>()?;
    m.add_class::<PyCommonAverageReference>()?;
    m.add_class::<PyMedianFilter>()?;
    m.add_class::<PyTeagerKaiser>()?;
    m.add_class::<PyTemplateFilter>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyDspSession>()?;
    m.add_class::<PyPca>()?;

    // Deep Learning (`dsp-synapse-ml`) & Burn-ONNX (`onnx-ir`) Classes
    m.add_class::<PySpatiotemporalUnetDenoiser>()?;
    m.add_class::<PySingleChannelDenoiser>()?;
    m.add_class::<PyDartsortVaeEmbedder>()?;
    m.add_class::<PyContrastiveWaveformEmbedder>()?;
    m.add_class::<PyUnitQualityClassifier>()?;
    m.add_class::<PyOnnxModelRunner>()?;

    // Direct Functions
    m.add_function(wrap_pyfunction!(list_nwb_series, m)?)?;
    m.add_function(wrap_pyfunction!(sort_recording, m)?)?;
    m.add_function(wrap_pyfunction!(notch_filter, m)?)?;
    m.add_function(wrap_pyfunction!(bandpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(common_average_reference, m)?)?;
    m.add_function(wrap_pyfunction!(scale_samples, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_9p, m)?)?;
    m.add_function(wrap_pyfunction!(teager_kaiser_filter, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_template, m)?)?;
    m.add_function(wrap_pyfunction!(detect_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_noise, m)?)?;
    m.add_function(wrap_pyfunction!(extract_snippets, m)?)?;
    m.add_function(wrap_pyfunction!(compute_isi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_snr, m)?)?;
    m.add_function(wrap_pyfunction!(compute_template, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[pymodule]
fn dsp_kitchen_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_bindings(m)
}
