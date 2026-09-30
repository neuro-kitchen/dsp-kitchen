use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array::to_numpy;
use dsp_synapse::metrics::{compute_isi_violations, compute_snr as rust_compute_snr, compute_mean_template};
use super::extraction::PyWaveformSnippet;

/// ISI violations (SpikeInterface `isi_violations`). `total_duration_sec` defaults to the span up to
/// the last spike, a lower bound of the recording duration: pass the real duration.
#[pyfunction]
#[pyo3(signature = (spike_samples, sample_rate_hz=30000.0, refractory_ms=1.5, total_duration_sec=None, min_isi_ms=0.0))]
pub fn compute_isi<'py>(
    py: Python<'py>,
    spike_samples: Vec<u64>,
    sample_rate_hz: f64,
    refractory_ms: f64,
    total_duration_sec: Option<f64>,
    min_isi_ms: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let duration = total_duration_sec
        .unwrap_or_else(|| spike_samples.iter().max().map_or(0.0, |&m| (m + 1) as f64 / sample_rate_hz));
    let res = compute_isi_violations(&spike_samples, sample_rate_hz, duration, refractory_ms, min_isi_ms);
    let dict = PyDict::new(py);
    dict.set_item("total_spikes", res.total_spikes)?;
    dict.set_item("violation_count", res.violation_count)?;
    dict.set_item("violation_rate_pct", res.violation_rate_pct)?;
    dict.set_item("isi_violations_ratio", res.isi_violations_ratio)?;
    dict.set_item("violations_per_sec", res.violations_per_sec)?;
    dict.set_item("firing_rate_hz", res.firing_rate_hz)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (peak_amplitude_uv, noise_std_uv))]
pub fn compute_snr(peak_amplitude_uv: f32, noise_std_uv: f32) -> f32 {
    rust_compute_snr(peak_amplitude_uv, noise_std_uv)
}

#[pyfunction]
#[pyo3(signature = (snippets))]
pub fn compute_template<'py>(
    py: Python<'py>,
    snippets: Vec<PyRef<PyWaveformSnippet>>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let rust_snippets: Vec<_> = snippets.iter().map(|s| s.inner.clone()).collect();
    let template = match compute_mean_template(&rust_snippets) {
        Some(t) => t,
        None => return Ok(None),
    };

    let (k, s) = (template.num_channels, template.num_samples);
    let mean_arr = to_numpy(py, template.mean, &[k, s])?;
    let std_arr = to_numpy(py, template.std, &[k, s])?;

    let dict = PyDict::new(py);
    dict.set_item("mean", mean_arr)?;
    dict.set_item("std", std_arr)?;
    dict.set_item("num_channels", k)?;
    dict.set_item("num_samples", s)?;
    Ok(Some(dict))
}
