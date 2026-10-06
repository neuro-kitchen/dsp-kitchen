//! Unit quality and response metrics. Defaults are SpikeInterface's (named in
//! `dsp_synapse::metrics`); analysis windows without a standard (rates, PSTH, MEP) are required.
//! Spike times are sample indices; `fs` (Hz) is always given.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::array::{to_numpy, F32Array};
use dsp_synapse::metrics::{
    compute_amplitude_cutoff as amplitude_cutoff, compute_autocorrelogram as autocorrelogram, compute_crosscorrelogram as crosscorrelogram,
    compute_d_prime as d_prime, compute_instantaneous_firing_rate, compute_isi_violations, compute_isolation_distance as isolation_distance,
    compute_mean_template, compute_presence_ratio as presence_ratio, compute_psth as psth, compute_silhouette_score as silhouette, compute_snr as snr,
    compute_stimulus_triggered_average, quantify_mep, Correlogram, DEFAULT_CORRELOGRAM_BIN_MS, DEFAULT_CORRELOGRAM_WINDOW_MS, DEFAULT_ISI_THRESHOLD_MS,
    DEFAULT_MIN_ISI_MS, DEFAULT_PRESENCE_BIN_SEC, DEFAULT_PRESENCE_MEAN_FR_RATIO,
};

use super::extraction::PyWaveformSnippet;

const MS_PER_S: f64 = 1e3;

fn seconds(samples: &[u64], fs: f64) -> Vec<f64> {
    samples.iter().map(|&s| s as f64 / fs).collect()
}

/// ISI violations (SpikeInterface `isi_violations`, Hill et al. ratio). `duration_sec` is the
/// recording's length (rates depend on it).
#[pyfunction]
#[pyo3(signature = (spike_samples, *, fs, duration_sec, isi_threshold_ms=DEFAULT_ISI_THRESHOLD_MS, min_isi_ms=DEFAULT_MIN_ISI_MS))]
pub fn compute_isi<'py>(py: Python<'py>, spike_samples: Vec<u64>, fs: f64, duration_sec: f64, isi_threshold_ms: f64, min_isi_ms: f64) -> PyResult<Bound<'py, PyDict>> {
    let res = compute_isi_violations(&spike_samples, fs, duration_sec, isi_threshold_ms, min_isi_ms);
    let dict = PyDict::new(py);
    dict.set_item("total_spikes", res.total_spikes)?;
    dict.set_item("violation_count", res.violation_count)?;
    dict.set_item("violation_rate_pct", res.violation_rate_pct)?;
    dict.set_item("isi_violations_ratio", res.isi_violations_ratio)?;
    dict.set_item("violations_per_sec", res.violations_per_sec)?;
    dict.set_item("firing_rate_hz", res.firing_rate_hz)?;
    Ok(dict)
}

/// Peak amplitude over noise σ (NaN when the noise is unknown or zero).
#[pyfunction]
pub fn compute_snr(peak_amplitude: f32, noise_std: f32) -> f32 {
    snr(peak_amplitude, noise_std)
}

/// Mean waveform of `snippets` (`[channels, samples]`), its SD (ddof 0, as Phy) and standard
/// error (from the ddof-1 SD; NaN below two spikes).
#[pyfunction]
pub fn compute_template<'py>(py: Python<'py>, snippets: Vec<PyRef<'py, PyWaveformSnippet>>) -> PyResult<Option<Bound<'py, PyDict>>> {
    let all: Vec<_> = snippets.iter().map(|s| s.inner.clone()).collect();
    let n = all.len();
    let Some(template) = compute_mean_template(&all) else { return Ok(None) };
    let (k, s) = (template.num_channels, template.num_samples);
    let se: Vec<f32> = template.std.iter().map(|&sd| if n < 2 { f32::NAN } else { sd / ((n - 1) as f32).sqrt() }).collect();
    let dict = PyDict::new(py);
    dict.set_item("mean", to_numpy(py, template.mean, &[k, s])?)?;
    dict.set_item("std", to_numpy(py, template.std, &[k, s])?)?;
    dict.set_item("se", to_numpy(py, se, &[k, s])?)?;
    dict.set_item("count", n)?;
    Ok(Some(dict))
}

fn correlogram_dict<'py>(py: Python<'py>, c: Correlogram) -> PyResult<Bound<'py, PyDict>> {
    let n = c.bin_centers_ms.len();
    let dict = PyDict::new(py);
    dict.set_item("bin_centers_ms", to_numpy(py, c.bin_centers_ms, &[n])?)?;
    dict.set_item("counts", c.counts)?;
    dict.set_item("bin_ms", c.bin_size_ms)?;
    dict.set_item("window_ms", c.window_ms)?;
    Ok(dict)
}

/// Autocorrelogram over ±`window_ms` in `bin_ms` bins.
#[pyfunction]
#[pyo3(signature = (spike_samples, *, fs, bin_ms=DEFAULT_CORRELOGRAM_BIN_MS, window_ms=DEFAULT_CORRELOGRAM_WINDOW_MS))]
pub fn compute_autocorrelogram<'py>(py: Python<'py>, mut spike_samples: Vec<u64>, fs: f64, bin_ms: f32, window_ms: f32) -> PyResult<Bound<'py, PyDict>> {
    spike_samples.sort_unstable();
    correlogram_dict(py, autocorrelogram(&spike_samples, fs, bin_ms, window_ms))
}

/// Cross-correlogram of `b` relative to `a` over ±`window_ms`.
#[pyfunction]
#[pyo3(signature = (spike_samples_a, spike_samples_b, *, fs, bin_ms=DEFAULT_CORRELOGRAM_BIN_MS, window_ms=DEFAULT_CORRELOGRAM_WINDOW_MS))]
pub fn compute_crosscorrelogram<'py>(py: Python<'py>, mut spike_samples_a: Vec<u64>, mut spike_samples_b: Vec<u64>, fs: f64, bin_ms: f32, window_ms: f32) -> PyResult<Bound<'py, PyDict>> {
    spike_samples_a.sort_unstable();
    spike_samples_b.sort_unstable();
    correlogram_dict(py, crosscorrelogram(&spike_samples_a, &spike_samples_b, fs, bin_ms, window_ms))
}

/// Firing rate (Hz) in `bin_ms` bins smoothed by a Gaussian of `sigma_ms`.
#[pyfunction]
#[pyo3(signature = (spike_samples, *, fs, duration_sec, bin_ms, sigma_ms))]
pub fn compute_firing_rate<'py>(py: Python<'py>, spike_samples: Vec<u64>, fs: f64, duration_sec: f64, bin_ms: f64, sigma_ms: f64) -> PyResult<Bound<'py, PyDict>> {
    let curve = compute_instantaneous_firing_rate(&seconds(&spike_samples, fs), duration_sec, bin_ms, sigma_ms);
    let n = curve.rate_hz.len();
    let dict = PyDict::new(py);
    dict.set_item("time_sec", curve.time_bin_centers_sec)?;
    dict.set_item("rate_hz", to_numpy(py, curve.rate_hz, &[n])?)?;
    Ok(dict)
}

/// Peri-stimulus time histogram: rate (Hz) and its standard error across trials, `pre_ms` before
/// to `post_ms` after each trigger in `bin_ms` bins.
#[pyfunction]
#[pyo3(signature = (spike_samples, trigger_samples, *, fs, pre_ms, post_ms, bin_ms))]
pub fn compute_psth<'py>(py: Python<'py>, spike_samples: Vec<u64>, trigger_samples: Vec<u64>, fs: f64, pre_ms: f64, post_ms: f64, bin_ms: f64) -> PyResult<Bound<'py, PyDict>> {
    let res = psth(&seconds(&spike_samples, fs), &seconds(&trigger_samples, fs), pre_ms, post_ms, bin_ms);
    let n = res.mean_rate_hz.len();
    let dict = PyDict::new(py);
    dict.set_item("bin_centers_ms", res.time_bins_ms)?;
    dict.set_item("rate_hz", to_numpy(py, res.mean_rate_hz, &[n])?)?;
    dict.set_item("se_rate_hz", to_numpy(py, res.se_rate_hz, &[n])?)?;
    dict.set_item("num_trials", res.num_trials)?;
    Ok(dict)
}

fn window_samples(ms: f64, fs: f64) -> usize {
    (ms / MS_PER_S * fs).round() as usize
}

/// Stimulus-triggered average of `data` (`[channels, samples]`): mean, SD (ddof 1) and standard
/// error, `pre_ms` before to `post_ms` after each trigger.
#[pyfunction]
#[pyo3(signature = (data, trigger_samples, *, fs, pre_ms, post_ms))]
pub fn compute_sta<'py>(py: Python<'py>, data: Bound<'py, PyAny>, trigger_samples: Vec<u64>, fs: f64, pre_ms: f64, post_ms: f64) -> PyResult<Bound<'py, PyDict>> {
    let input = F32Array::new(&data)?;
    let (channels, samples) = input.channels_samples(None)?;
    let x = input.slice();
    let sta = py.detach(|| compute_stimulus_triggered_average(x, channels, samples, &trigger_samples, window_samples(pre_ms, fs), window_samples(post_ms, fs), fs));
    let w = sta.num_samples;
    let dict = PyDict::new(py);
    dict.set_item("mean", to_numpy(py, sta.mean_uv, &[channels, w])?)?;
    dict.set_item("std", to_numpy(py, sta.std_uv, &[channels, w])?)?;
    dict.set_item("se", to_numpy(py, sta.se_uv, &[channels, w])?)?;
    dict.set_item("time_ms", sta.time_ms)?;
    dict.set_item("num_trials", sta.num_trials)?;
    Ok(dict)
}

/// Motor evoked potentials of `data` (`[channels, samples]`) after `trigger_samples`: the
/// stimulus-triggered average, then per channel the onset (first crossing of
/// `threshold_sigma` · pre-stimulus SD within `response_window_ms`), peak-to-peak, RMS and
/// rectified area. Onsets that never cross are NaN.
#[pyfunction]
#[pyo3(signature = (data, trigger_samples, *, fs, pre_ms, post_ms, response_window_ms, threshold_sigma))]
#[allow(clippy::too_many_arguments)]
pub fn compute_mep<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    trigger_samples: Vec<u64>,
    fs: f64,
    pre_ms: f64,
    post_ms: f64,
    response_window_ms: (f64, f64),
    threshold_sigma: f32,
) -> PyResult<Bound<'py, PyList>> {
    let input = F32Array::new(&data)?;
    let (channels, samples) = input.channels_samples(None)?;
    let x = input.slice();
    let meps = py.detach(|| {
        let sta = compute_stimulus_triggered_average(x, channels, samples, &trigger_samples, window_samples(pre_ms, fs), window_samples(post_ms, fs), fs);
        quantify_mep(&sta, response_window_ms.0, response_window_ms.1, threshold_sigma)
    });
    let out = PyList::empty(py);
    for m in meps {
        let d = PyDict::new(py);
        d.set_item("channel", m.channel_id)?;
        d.set_item("onset_latency_ms", m.onset_latency_ms)?;
        d.set_item("peak_to_peak", m.peak_to_peak_uv)?;
        d.set_item("rms", m.rms_uv)?;
        d.set_item("rectified_auc_ms", m.rectified_auc_uv_ms)?;
        out.append(d)?;
    }
    Ok(out)
}

/// `(values, rows, columns)` of a 2-D array.
fn rows<'a>(a: &'a F32Array<'_>, what: &str) -> PyResult<(&'a [f32], usize, usize)> {
    match *a.shape() {
        [n, d] => Ok((a.slice(), n, d)),
        _ => Err(PyValueError::new_err(format!("{what} must be a 2-D [spikes, features] array"))),
    }
}

/// Fisher discriminant d′ between two clusters of features (`[n_a, d]`, `[n_b, d]`).
#[pyfunction]
pub fn compute_d_prime(cluster_a: Bound<'_, PyAny>, cluster_b: Bound<'_, PyAny>) -> PyResult<f32> {
    let (a, b) = (F32Array::new(&cluster_a)?, F32Array::new(&cluster_b)?);
    let ((xa, na, da), (xb, nb, db)) = (rows(&a, "cluster_a")?, rows(&b, "cluster_b")?);
    if da != db {
        return Err(PyValueError::new_err("clusters must have the same number of features"));
    }
    Ok(d_prime(xa, na, xb, nb, da))
}

/// Isolation distance (Schmitzer-Torbert et al. 2005) of `target_unit`.
#[pyfunction]
pub fn compute_isolation_distance(features: Bound<'_, PyAny>, labels: Vec<usize>, target_unit: usize) -> PyResult<f32> {
    let f = F32Array::new(&features)?;
    let (x, n, d) = rows(&f, "features")?;
    if n != labels.len() {
        return Err(PyValueError::new_err("one label per row of features"));
    }
    Ok(isolation_distance(x, &labels, d, target_unit))
}

/// Silhouette score in `[-1, 1]` of `target_unit`.
#[pyfunction]
pub fn compute_silhouette_score(features: Bound<'_, PyAny>, labels: Vec<usize>, target_unit: usize) -> PyResult<f32> {
    let f = F32Array::new(&features)?;
    let (x, n, d) = rows(&f, "features")?;
    if n != labels.len() {
        return Err(PyValueError::new_err("one label per row of features"));
    }
    Ok(silhouette(x, &labels, d, target_unit))
}

/// Amplitude cutoff (IBL / SpikeInterface): estimated fraction of spikes below detection, in
/// `[0, 0.5]`.
#[pyfunction]
pub fn compute_amplitude_cutoff(amplitudes: Vec<f32>) -> f64 {
    amplitude_cutoff(&amplitudes)
}

/// Fraction of `bin_duration_sec` bins in which the unit fires above `mean_fr_ratio` × its mean
/// rate (SpikeInterface `presence_ratio`).
#[pyfunction]
#[pyo3(signature = (spike_samples, total_samples, *, fs, bin_duration_sec=DEFAULT_PRESENCE_BIN_SEC, mean_fr_ratio=DEFAULT_PRESENCE_MEAN_FR_RATIO))]
pub fn compute_presence_ratio(spike_samples: Vec<u64>, total_samples: u64, fs: f64, bin_duration_sec: f64, mean_fr_ratio: f64) -> f64 {
    presence_ratio(&spike_samples, total_samples, fs, bin_duration_sec, mean_fr_ratio)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_isi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_snr, m)?)?;
    m.add_function(wrap_pyfunction!(compute_template, m)?)?;
    m.add_function(wrap_pyfunction!(compute_autocorrelogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_crosscorrelogram, m)?)?;
    m.add_function(wrap_pyfunction!(compute_firing_rate, m)?)?;
    m.add_function(wrap_pyfunction!(compute_psth, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sta, m)?)?;
    m.add_function(wrap_pyfunction!(compute_mep, m)?)?;
    m.add_function(wrap_pyfunction!(compute_d_prime, m)?)?;
    m.add_function(wrap_pyfunction!(compute_isolation_distance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(compute_amplitude_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(compute_presence_ratio, m)?)?;
    Ok(())
}
