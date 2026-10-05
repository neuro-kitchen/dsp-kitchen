pub mod array;
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
    bandpass_filter, highpass_filter, lowpass_filter, median_filter_9p, notch_filter,
    subtract_template, teager_kaiser_filter, PyBandpassFilter, PyBandstopFilter,
    PyHighpassFilter, PyLowpassFilter, PyMedianFilter, PyNotchFilter, PyTeagerKaiser,
    PyTemplateFilter,
};
use crate::linalg::{PyFastIca, PyPca, PyPpca};
use crate::math::{scale_samples, PyClamp, PyScale, PySubtractBaseline};
use crate::pipeline::{PyDspSession, PyPipeline};
use crate::spatial::{
    common_average_reference, PyCommonAverageReference, PySpatialWhitening, PySurfaceLaplacian,
};
use crate::synapse::{
    cluster_density_peaks, cluster_gmm, cluster_isosplit, compare_sortings,
    compare_spike_trains, compute_amplitude_cutoff, compute_autocorrelogram,
    compute_crosscorrelogram, compute_d_prime, compute_firing_rate, compute_isi,
    compute_isolation_distance, compute_presence_ratio, compute_psth,
    compute_silhouette_score, compute_snr, compute_sta, compute_template,
    correct_drift_kriging, decompose_hdemg_cbss, deduplicate_spikes, detect_spikes,
    estimate_noise, estimate_nonrigid_drift, estimate_rigid_drift, export_to_phy,
    extract_snippets, load_nwb_units, load_sorting, localize_spikes, match_spikes_omp,
    quantify_mep, read_kilosort, save_nwb_units, save_sorting, sort_recording,
    PyDeduplicatedSpike, PyProbeLayout, PySortingOutput, PySpikeEvent,
    PyStreamingSortResult, PyWaveformSnippet,
};
use crate::synapse_ml::{
    PyEmusortBasisEmbedder, PyEmusortDetector, PyEmusortLatencyAligner, PyEmusortSortConfig,
    PyKilosort4BasisEmbedder, PyKilosort4Detector, PyModelHub,
};

fn register_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core & Classical Classes
    m.add_class::<PyProbeLayout>()?;
    m.add_class::<PySpikeEvent>()?;
    m.add_class::<PyDeduplicatedSpike>()?;
    m.add_class::<PyWaveformSnippet>()?;
    m.add_class::<PyStreamingSortResult>()?;
    m.add_class::<PySortingOutput>()?;
    m.add_class::<PyMmapRecording>()?;
    m.add_class::<PyNwbZarrRecording>()?;
    m.add_class::<PyScale>()?;
    m.add_class::<PySubtractBaseline>()?;
    m.add_class::<PyClamp>()?;
    m.add_class::<PyNotchFilter>()?;
    m.add_class::<PyBandpassFilter>()?;
    m.add_class::<PyHighpassFilter>()?;
    m.add_class::<PyLowpassFilter>()?;
    m.add_class::<PyBandstopFilter>()?;
    m.add_class::<PyCommonAverageReference>()?;
    m.add_class::<PySpatialWhitening>()?;
    m.add_class::<PySurfaceLaplacian>()?;
    m.add_class::<PyMedianFilter>()?;
    m.add_class::<PyTeagerKaiser>()?;
    m.add_class::<PyTemplateFilter>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyDspSession>()?;
    m.add_class::<PyPca>()?;
    m.add_class::<PyPpca>()?;
    m.add_class::<PyFastIca>()?;

    // Pretrained Model Hub & Models (`dsp-synapse-ml`)
    m.add_class::<PyModelHub>()?;
    m.add_class::<PyKilosort4BasisEmbedder>()?;
    m.add_class::<PyKilosort4Detector>()?;
    m.add_class::<PyEmusortSortConfig>()?;
    m.add_class::<PyEmusortDetector>()?;
    m.add_class::<PyEmusortBasisEmbedder>()?;
    m.add_class::<PyEmusortLatencyAligner>()?;
    m.add("MyomatrixSortConfig", m.getattr("EmusortSortConfig")?)?;
    m.add("MyomatrixDetector", m.getattr("EmusortDetector")?)?;
    m.add("MyomatrixBasisEmbedder", m.getattr("EmusortBasisEmbedder")?)?;
    m.add("MyomatrixLatencyAligner", m.getattr("EmusortLatencyAligner")?)?;

    // Direct Functions
    m.add_function(wrap_pyfunction!(list_nwb_series, m)?)?;
    m.add_function(wrap_pyfunction!(sort_recording, m)?)?;
    m.add_function(wrap_pyfunction!(notch_filter, m)?)?;
    m.add_function(wrap_pyfunction!(bandpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(highpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(lowpass_filter, m)?)?;
    m.add_function(wrap_pyfunction!(common_average_reference, m)?)?;
    m.add_function(wrap_pyfunction!(scale_samples, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_9p, m)?)?;
    m.add_function(wrap_pyfunction!(teager_kaiser_filter, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_template, m)?)?;
    m.add_function(wrap_pyfunction!(detect_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_noise, m)?)?;
    m.add_function(wrap_pyfunction!(extract_snippets, m)?)?;
    m.add_function(wrap_pyfunction!(localize_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_rigid_drift, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_nonrigid_drift, m)?)?;
    m.add_function(wrap_pyfunction!(correct_drift_kriging, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_gmm, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_density_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_isosplit, m)?)?;
    m.add_function(wrap_pyfunction!(match_spikes_omp, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_hdemg_cbss, m)?)?;
    m.add_function(wrap_pyfunction!(compute_isi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_snr, m)?)?;
    m.add_function(wrap_pyfunction!(compute_template, m)?)?;
    m.add_function(wrap_pyfunction!(compute_autocorrelogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_crosscorrelogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_firing_rate, m)?)?;
    m.add_function(wrap_pyfunction!(compute_psth, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sta, m)?)?;
    m.add_function(wrap_pyfunction!(quantify_mep, m)?)?;
    m.add_function(wrap_pyfunction!(compute_d_prime, m)?)?;
    m.add_function(wrap_pyfunction!(compute_isolation_distance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_amplitude_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(compute_presence_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(save_sorting, m)?)?;
    m.add_function(wrap_pyfunction!(load_sorting, m)?)?;
    m.add_function(wrap_pyfunction!(export_to_phy, m)?)?;
    m.add_function(wrap_pyfunction!(read_kilosort, m)?)?;
    m.add_function(wrap_pyfunction!(save_nwb_units, m)?)?;
    m.add_function(wrap_pyfunction!(load_nwb_units, m)?)?;
    m.add_function(wrap_pyfunction!(compare_sortings, m)?)?;
    m.add_function(wrap_pyfunction!(compare_spike_trains, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[pymodule]
fn dsp_kitchen_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_bindings(m)
}
