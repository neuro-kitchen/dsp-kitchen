use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array::{to_numpy, F32Array};
use dsp_synapse::metrics::{
    compute_amplitude_cutoff as rust_compute_amplitude_cutoff,
    compute_autocorrelogram as rust_compute_autocorrelogram,
    compute_crosscorrelogram as rust_compute_crosscorrelogram,
    compute_d_prime as rust_compute_d_prime,
    compute_instantaneous_firing_rate, compute_isi_violations,
    compute_isolation_distance as rust_compute_isolation_distance, compute_mean_template,
    compute_presence_ratio as rust_compute_presence_ratio, compute_psth as rust_compute_psth,
    compute_silhouette_score as rust_compute_silhouette_score, compute_snr as rust_compute_snr,
    compute_stimulus_triggered_average, quantify_mep as rust_quantify_mep,
    StimulusTriggeredAverage,
};
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
    let duration = total_duration_sec.unwrap_or_else(|| {
        spike_samples
            .iter()
            .max()
            .map_or(0.0, |&m| (m + 1) as f64 / sample_rate_hz)
    });
    let res = compute_isi_violations(
        &spike_samples,
        sample_rate_hz,
        duration,
        refractory_ms,
        min_isi_ms,
    );
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
    let n = rust_snippets.len();
    let template = match compute_mean_template(&rust_snippets) {
        Some(t) => t,
        None => return Ok(None),
    };

    let (k, s) = (template.num_channels, template.num_samples);
    let inv_sqrt_n = 1.0 / (n.max(1) as f32).sqrt();
    let se_vec: Vec<f32> = template.std.iter().map(|&sd| sd * inv_sqrt_n).collect();

    let mean_arr = to_numpy(py, template.mean, &[k, s])?;
    let std_arr = to_numpy(py, template.std, &[k, s])?;
    let se_arr = to_numpy(py, se_vec, &[k, s])?;

    let dict = PyDict::new(py);
    dict.set_item("mean", mean_arr)?;
    dict.set_item("std", std_arr)?;
    dict.set_item("se", se_arr)?;
    dict.set_item("count", n)?;
    dict.set_item("num_channels", k)?;
    dict.set_item("num_samples", s)?;
    Ok(Some(dict))
}

/// Fast $O(N)$ symmetric Auto-Correlogram (ACG) over `[-window_ms, +window_ms]`.
#[pyfunction]
#[pyo3(signature = (spike_samples, sample_rate_hz=30000.0, bin_size_ms=1.0, window_ms=50.0))]
pub fn compute_autocorrelogram<'py>(
    py: Python<'py>,
    mut spike_samples: Vec<u64>,
    sample_rate_hz: f64,
    bin_size_ms: f32,
    window_ms: f32,
) -> PyResult<Bound<'py, PyDict>> {
    spike_samples.sort_unstable();
    let res = rust_compute_autocorrelogram(&spike_samples, sample_rate_hz, bin_size_ms, window_ms);
    let n_bins = res.bin_centers_ms.len();
    let dict = PyDict::new(py);
    dict.set_item(
        "bin_centers_ms",
        to_numpy(py, res.bin_centers_ms, &[n_bins])?,
    )?;
    dict.set_item("counts", res.counts)?;
    dict.set_item("bin_size_ms", res.bin_size_ms)?;
    dict.set_item("window_ms", res.window_ms)?;
    Ok(dict)
}

/// Fast $O(N)$ Cross-Correlogram (CCG) between `spike_samples_a` and `spike_samples_b`.
#[pyfunction]
#[pyo3(signature = (spike_samples_a, spike_samples_b, sample_rate_hz=30000.0, bin_size_ms=1.0, window_ms=50.0))]
pub fn compute_crosscorrelogram<'py>(
    py: Python<'py>,
    mut spike_samples_a: Vec<u64>,
    mut spike_samples_b: Vec<u64>,
    sample_rate_hz: f64,
    bin_size_ms: f32,
    window_ms: f32,
) -> PyResult<Bound<'py, PyDict>> {
    spike_samples_a.sort_unstable();
    spike_samples_b.sort_unstable();
    let res = rust_compute_crosscorrelogram(
        &spike_samples_a,
        &spike_samples_b,
        sample_rate_hz,
        bin_size_ms,
        window_ms,
    );
    let n_bins = res.bin_centers_ms.len();
    let dict = PyDict::new(py);
    dict.set_item(
        "bin_centers_ms",
        to_numpy(py, res.bin_centers_ms, &[n_bins])?,
    )?;
    dict.set_item("counts", res.counts)?;
    dict.set_item("bin_size_ms", res.bin_size_ms)?;
    dict.set_item("window_ms", res.window_ms)?;
    Ok(dict)
}

/// Continuous Gaussian-smoothed instantaneous firing rate curve $r(t)$ (in Hz).
#[pyfunction]
#[pyo3(signature = (spike_samples, total_duration_sec, sample_rate_hz=30000.0, bin_dt_sec=0.01, sigma_ms=25.0))]
pub fn compute_firing_rate<'py>(
    py: Python<'py>,
    spike_samples: Vec<u64>,
    total_duration_sec: f64,
    sample_rate_hz: f64,
    bin_dt_sec: f64,
    sigma_ms: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let fs = sample_rate_hz.max(1.0);
    let spike_times_sec: Vec<f64> = spike_samples.iter().map(|&s| (s as f64) / fs).collect();
    let curve = compute_instantaneous_firing_rate(
        &spike_times_sec,
        total_duration_sec,
        bin_dt_sec * 1000.0,
        sigma_ms,
    );
    let n = curve.rate_hz.len();
    let dict = PyDict::new(py);
    dict.set_item("time_sec", curve.time_bin_centers_sec)?;
    dict.set_item("rate_hz", to_numpy(py, curve.rate_hz, &[n])?)?;
    dict.set_item("bin_dt_sec", curve.bin_width_sec)?;
    dict.set_item("sigma_ms", curve.kernel_sigma_ms)?;
    Ok(dict)
}

/// Peri-Stimulus Time Histogram (PSTH) aligned to `trigger_samples`.
#[pyfunction]
#[pyo3(signature = (spike_samples, trigger_samples, sample_rate_hz=30000.0, pre_ms=50.0, post_ms=100.0, bin_ms=2.0, smooth_sigma_ms=0.0))]
pub fn compute_psth<'py>(
    py: Python<'py>,
    spike_samples: Vec<u64>,
    trigger_samples: Vec<u64>,
    sample_rate_hz: f64,
    pre_ms: f64,
    post_ms: f64,
    bin_ms: f64,
    smooth_sigma_ms: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = smooth_sigma_ms;
    let fs = sample_rate_hz.max(1.0);
    let spike_times_sec: Vec<f64> = spike_samples.iter().map(|&s| (s as f64) / fs).collect();
    let stim_times_sec: Vec<f64> = trigger_samples.iter().map(|&s| (s as f64) / fs).collect();
    let res = rust_compute_psth(&spike_times_sec, &stim_times_sec, pre_ms, post_ms, bin_ms);
    let n = res.mean_rate_hz.len();
    let time_ms_f32: Vec<f32> = res.time_bins_ms.iter().map(|&t| t as f32).collect();
    let dict = PyDict::new(py);
    dict.set_item("bin_centers_ms", to_numpy(py, time_ms_f32, &[n])?)?;
    dict.set_item("rate_hz", to_numpy(py, res.mean_rate_hz, &[n])?)?;
    dict.set_item("se_rate_hz", to_numpy(py, res.se_rate_hz, &[n])?)?;
    dict.set_item("num_trials", res.num_trials)?;
    Ok(dict)
}

/// Stimulus-Triggered Average (STA) across `trigger_samples` with mean, SD, and SE ($\text{SD}/\sqrt{N}$).
#[pyfunction]
#[pyo3(signature = (data, trigger_samples, sample_rate_hz=30000.0, pre_ms=10.0, post_ms=50.0, channels=None))]
pub fn compute_sta<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    trigger_samples: Vec<u64>,
    sample_rate_hz: f64,
    pre_ms: f64,
    post_ms: f64,
    channels: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let input = F32Array::new(&data)?;
    let (ch, samples) = input.channels_samples(channels)?;
    let pre_samples = ((pre_ms * 1e-3 * sample_rate_hz).round() as usize).max(1);
    let post_samples = ((post_ms * 1e-3 * sample_rate_hz).round() as usize).max(1);
    let sta = compute_stimulus_triggered_average(
        input.slice(),
        ch,
        samples,
        &trigger_samples,
        pre_samples,
        post_samples,
        sample_rate_hz,
    );

    let w = sta.num_samples;
    let time_ms_f32: Vec<f32> = sta.time_ms.iter().map(|&t| t as f32).collect();
    let dict = PyDict::new(py);
    dict.set_item("mean", to_numpy(py, sta.mean_uv, &[ch, w])?)?;
    dict.set_item("std", to_numpy(py, sta.std_uv, &[ch, w])?)?;
    dict.set_item("se", to_numpy(py, sta.se_uv, &[ch, w])?)?;
    dict.set_item("time_ms", to_numpy(py, time_ms_f32, &[w])?)?;
    dict.set_item("num_trials", sta.num_trials)?;
    dict.set_item("num_channels", sta.num_channels)?;
    dict.set_item("num_window_samples", w)?;
    Ok(dict)
}

/// Quantify Motor Evoked Potential (MEP) onset latency, peak-to-peak amplitude, rectified AUC, and RMS.
#[pyfunction]
#[pyo3(signature = (waveform, time_ms, baseline_window_ms=(-10.0, -1.0), response_window_ms=(2.0, 45.0), threshold_sd=3.0))]
pub fn quantify_mep<'py>(
    py: Python<'py>,
    waveform: Bound<'py, PyAny>,
    time_ms: Vec<f32>,
    baseline_window_ms: (f32, f32),
    response_window_ms: (f32, f32),
    threshold_sd: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = baseline_window_ms;
    let wave_arr = F32Array::new(&waveform)?;
    let w_slice = wave_arr.slice();
    let n = w_slice.len().min(time_ms.len());
    let sta = StimulusTriggeredAverage {
        num_channels: 1,
        num_samples: n,
        num_trials: 1,
        time_ms: time_ms[..n].iter().map(|&t| t as f64).collect(),
        mean_uv: w_slice[..n].to_vec(),
        std_uv: vec![0.0; n],
        se_uv: vec![0.0; n],
    };
    let meps = rust_quantify_mep(
        &sta,
        response_window_ms.0 as f64,
        response_window_ms.1 as f64,
        threshold_sd,
    );
    let dict = PyDict::new(py);
    if let Some(m) = meps.first() {
        let onset = if m.onset_latency_ms.is_nan() {
            None
        } else {
            Some(m.onset_latency_ms)
        };
        dict.set_item("onset_latency_ms", onset)?;
        dict.set_item("peak_to_peak_uv", m.peak_to_peak_uv)?;
        dict.set_item("rectified_auc_uv_ms", m.rectified_auc_uv_ms)?;
        dict.set_item("rms_uv", m.rms_uv)?;
    }
    Ok(dict)
}

/// Fisher Linear Discriminant sensitivity $d'$ between two feature clusters `[N_a, D]` and `[N_b, D]`.
#[pyfunction]
pub fn compute_d_prime<'py>(
    cluster_a: Bound<'py, PyAny>,
    cluster_b: Bound<'py, PyAny>,
) -> PyResult<f32> {
    let a = F32Array::new(&cluster_a)?;
    let b = F32Array::new(&cluster_b)?;
    if a.shape().len() != 2 || b.shape().len() != 2 || a.shape()[1] != b.shape()[1] {
        return Err(PyValueError::new_err(
            "cluster_a and cluster_b must be 2D arrays with matching feature dimension D",
        ));
    }
    Ok(rust_compute_d_prime(
        a.slice(),
        a.shape()[0],
        b.slice(),
        b.shape()[0],
        a.shape()[1],
    ))
}

/// Mahalanobis Isolation Distance (Schmitzer-Torbert et al. 2005) for `target_unit`.
#[pyfunction]
pub fn compute_isolation_distance<'py>(
    features: Bound<'py, PyAny>,
    labels: Vec<usize>,
    target_unit: usize,
) -> PyResult<f32> {
    let f = F32Array::new(&features)?;
    if f.shape().len() != 2 || f.shape()[0] != labels.len() {
        return Err(PyValueError::new_err(
            "features must have shape [len(labels), dim]",
        ));
    }
    Ok(rust_compute_isolation_distance(
        f.slice(),
        &labels,
        f.shape()[1],
        target_unit,
    ))
}

/// Silhouette score in `[-1, 1]` for `target_unit`.
#[pyfunction]
pub fn compute_silhouette_score<'py>(
    features: Bound<'py, PyAny>,
    labels: Vec<usize>,
    target_unit: usize,
) -> PyResult<f32> {
    let f = F32Array::new(&features)?;
    if f.shape().len() != 2 || f.shape()[0] != labels.len() {
        return Err(PyValueError::new_err(
            "features must have shape [len(labels), dim]",
        ));
    }
    Ok(rust_compute_silhouette_score(
        f.slice(),
        &labels,
        f.shape()[1],
        target_unit,
    ))
}

/// IBL / SpikeInterface amplitude cutoff quality metric in `[0.0, 0.5]`.
#[pyfunction]
pub fn compute_amplitude_cutoff(amplitudes: Vec<f32>) -> f64 {
    rust_compute_amplitude_cutoff(&amplitudes)
}

/// Fraction of `bin_duration_s` bins in which the unit fires above `mean_fr_ratio_thresh` × its mean rate.
#[pyfunction]
#[pyo3(signature = (spike_samples, total_samples, sample_rate_hz=30000.0, bin_duration_s=60.0, mean_fr_ratio_thresh=0.0))]
pub fn compute_presence_ratio(
    spike_samples: Vec<u64>,
    total_samples: u64,
    sample_rate_hz: f64,
    bin_duration_s: f64,
    mean_fr_ratio_thresh: f64,
) -> f64 {
    rust_compute_presence_ratio(
        &spike_samples,
        total_samples,
        sample_rate_hz,
        bin_duration_s,
        mean_fr_ratio_thresh,
    )
}
