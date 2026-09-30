use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use dsp_synapse::metrics::{compute_isi_violations, compute_snr as rust_compute_snr, compute_mean_template};
use super::extraction::PyWaveformSnippet;

#[pyfunction]
#[pyo3(signature = (spike_samples, sample_rate_hz=30000.0, refractory_ms=1.5))]
pub fn compute_isi<'py>(
    py: Python<'py>,
    spike_samples: Vec<u64>,
    sample_rate_hz: f64,
    refractory_ms: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let res = compute_isi_violations(&spike_samples, sample_rate_hz, refractory_ms);
    let dict = PyDict::new(py);
    dict.set_item("total_spikes", res.total_spikes)?;
    dict.set_item("violation_count", res.violation_count)?;
    dict.set_item("violation_rate_pct", res.violation_rate_pct)?;
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

    let np = py.import("numpy")?;
    let k = template.num_channels;
    let s = template.num_samples;

    let mean_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            template.mean.as_ptr() as *const u8,
            template.mean.len() * std::mem::size_of::<f32>(),
        )
    };
    let mean_py_bytes = PyBytes::new(py, mean_bytes);
    let mean_flat = np.call_method1("frombuffer", (mean_py_bytes, "float32"))?;
    let mean_arr = mean_flat.call_method1("reshape", ((k, s),))?;

    let std_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            template.std.as_ptr() as *const u8,
            template.std.len() * std::mem::size_of::<f32>(),
        )
    };
    let std_py_bytes = PyBytes::new(py, std_bytes);
    let std_flat = np.call_method1("frombuffer", (std_py_bytes, "float32"))?;
    let std_arr = std_flat.call_method1("reshape", ((k, s),))?;

    let dict = PyDict::new(py);
    dict.set_item("mean", mean_arr)?;
    dict.set_item("std", std_arr)?;
    dict.set_item("num_channels", k)?;
    dict.set_item("num_samples", s)?;
    Ok(Some(dict))
}
