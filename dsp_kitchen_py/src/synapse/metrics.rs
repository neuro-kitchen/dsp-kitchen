//! Unit quality and response metrics. Defaults are SpikeInterface's (named in
//! `dsp_synapse::metrics`); analysis windows without a standard (rates, PSTH, MEP) are required.
//! Spike times are sample indices; `fs` (Hz) is always given.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyfunction};
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

/// Inter-spike-interval violations of a unit (SpikeInterface `isi_violations`; Hill et al. ratio).
///
/// Parameters
/// ----------
/// spike_samples : list of int
///     Spike times, recording samples.
/// fs : float
///     Sampling rate, Hz.
/// duration_sec : float
///     Length of the recording, s (rates depend on it).
/// isi_threshold_ms : float, default 1.5
///     Intervals shorter than this violate the refractory period, ms.
/// min_isi_ms : float, default 0.0
///     Shortest possible interval (e.g. a duplicate-removal window), ms.
///
/// Returns
/// -------
/// dict
///     `total_spikes`, `violation_count`, `violation_rate_pct` (% of intervals),
///     `isi_violations_ratio` (Hill et al.: estimated contamination rate relative to the unit's rate),
///     `violations_per_sec`, `firing_rate_hz`.
#[gen_stub_pyfunction]
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

/// Signal-to-noise ratio: peak amplitude over noise σ.
///
/// Parameters
/// ----------
/// peak_amplitude : float
///     Peak of the unit's template, in the data's unit.
/// noise_std : float
///     Noise σ of the channel, same unit.
///
/// Returns
/// -------
/// float
///     `|peak_amplitude| / noise_std`; NaN when the noise is unknown or zero.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn compute_snr(peak_amplitude: f32, noise_std: f32) -> f32 {
    snr(peak_amplitude, noise_std)
}

/// Mean waveform of a unit's snippets, with its spread.
///
/// Parameters
/// ----------
/// snippets : list of WaveformSnippet
///     Snippets of one unit, all with the same channels and length.
///
/// Returns
/// -------
/// dict or None
///     `mean`, `std` (ddof 0, as Phy), `se` (standard error from the ddof-1 SD; NaN below two snippets):
///     `[channels, samples]` float32; `count`. `None` without snippets.
#[gen_stub_pyfunction]
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

/// Autocorrelogram of a unit: counts of the time differences between every pair of its spikes (a
/// spike is not paired with itself), in bins; symmetric around 0.
///
/// Parameters
/// ----------
/// spike_samples : list of int
///     Spike times, recording samples.
/// fs : float
///     Sampling rate, Hz.
/// bin_ms : float, default 1.0
///     Bin width, ms.
/// window_ms : float, default 50.0
///     Half width, ms: lags in `[-window_ms, window_ms]`.
///
/// Returns
/// -------
/// dict
///     `bin_centers_ms` (float32 array), `counts` (list of int), `bin_ms`, `window_ms`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (spike_samples, *, fs, bin_ms=DEFAULT_CORRELOGRAM_BIN_MS, window_ms=DEFAULT_CORRELOGRAM_WINDOW_MS))]
pub fn compute_autocorrelogram<'py>(py: Python<'py>, mut spike_samples: Vec<u64>, fs: f64, bin_ms: f32, window_ms: f32) -> PyResult<Bound<'py, PyDict>> {
    spike_samples.sort_unstable();
    correlogram_dict(py, autocorrelogram(&spike_samples, fs, bin_ms, window_ms))
}

/// Cross-correlogram of unit B relative to unit A: counts of `t_B − t_A` in bins (positive lags:
/// B after A). A refractory dip at 0 suggests the two are one neuron.
///
/// Parameters
/// ----------
/// spike_samples_a, spike_samples_b : list of int
///     Spike times of the two units, recording samples.
/// fs : float
///     Sampling rate, Hz.
/// bin_ms : float, default 1.0
///     Bin width, ms.
/// window_ms : float, default 50.0
///     Half width, ms.
///
/// Returns
/// -------
/// dict
///     `bin_centers_ms` (float32 array), `counts` (list of int), `bin_ms`, `window_ms`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (spike_samples_a, spike_samples_b, *, fs, bin_ms=DEFAULT_CORRELOGRAM_BIN_MS, window_ms=DEFAULT_CORRELOGRAM_WINDOW_MS))]
pub fn compute_crosscorrelogram<'py>(py: Python<'py>, mut spike_samples_a: Vec<u64>, mut spike_samples_b: Vec<u64>, fs: f64, bin_ms: f32, window_ms: f32) -> PyResult<Bound<'py, PyDict>> {
    spike_samples_a.sort_unstable();
    spike_samples_b.sort_unstable();
    correlogram_dict(py, crosscorrelogram(&spike_samples_a, &spike_samples_b, fs, bin_ms, window_ms))
}

/// Firing rate over time: spike counts in bins, smoothed by a Gaussian.
///
/// Parameters
/// ----------
/// spike_samples : list of int
///     Spike times, recording samples.
/// fs : float
///     Sampling rate, Hz.
/// duration_sec : float
///     Length of the recording, s.
/// bin_ms : float
///     Bin width, ms.
/// sigma_ms : float
///     Standard deviation of the smoothing Gaussian, ms.
///
/// Returns
/// -------
/// dict
///     `time_sec` (bin centres, s), `rate_hz` (float32 array, Hz).
#[gen_stub_pyfunction]
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

/// Peri-stimulus time histogram: the unit's rate around each trigger, averaged over trials.
///
/// Parameters
/// ----------
/// spike_samples : list of int
///     Spike times, recording samples.
/// trigger_samples : list of int
///     Stimulus times, recording samples (one trial each).
/// fs : float
///     Sampling rate, Hz.
/// pre_ms, post_ms : float
///     Window before and after each trigger, ms.
/// bin_ms : float
///     Bin width, ms.
///
/// Returns
/// -------
/// dict
///     `bin_centers_ms`, `rate_hz` (mean over trials, Hz), `se_rate_hz` (standard error across trials),
///     `num_trials`.
#[gen_stub_pyfunction]
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

/// Stimulus-triggered average of a signal around each trigger.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]`, converted to float32.
/// trigger_samples : list of int
///     Stimulus times, samples of `data`.
/// fs : float
///     Sampling rate, Hz.
/// pre_ms, post_ms : float
///     Window before and after each trigger, ms.
///
/// Returns
/// -------
/// dict
///     `mean`, `std` (ddof 1), `se`: `[channels, window samples]` float32 in the data's unit;
///     `time_ms` (relative to the trigger); `num_trials` (triggers whose window fits the data).
#[gen_stub_pyfunction]
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

/// Motor evoked potentials: the stimulus-triggered average, then per channel its onset, size and area.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     `[channels, samples]` (e.g. EMG), converted to float32.
/// trigger_samples : list of int
///     Stimulus times, samples of `data`.
/// fs : float
///     Sampling rate, Hz.
/// pre_ms, post_ms : float
///     Window before and after each trigger, ms (the pre-stimulus part gives the baseline SD).
/// response_window_ms : tuple of float
///     `(start, end)` after the trigger where the response is searched, ms.
/// threshold_sigma : float
///     The onset is the first crossing of `threshold_sigma` · pre-stimulus SD in the response window.
///
/// Returns
/// -------
/// list of dict
///     One per channel: `channel`, `onset_latency_ms` (NaN when it never crosses), `peak_to_peak`,
///     `rms` (data's unit), `rectified_auc_ms` (area of `|x|`, unit · ms).
#[gen_stub_pyfunction]
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

/// Fisher discriminant d′ between two clusters of features: how separated they are along the
/// direction that best separates them.
///
/// Parameters
/// ----------
/// cluster_a, cluster_b : numpy.ndarray
///     `[spikes, features]` of each cluster (same number of features).
///
/// Returns
/// -------
/// float
#[gen_stub_pyfunction]
#[pyfunction]
pub fn compute_d_prime(cluster_a: Bound<'_, PyAny>, cluster_b: Bound<'_, PyAny>) -> PyResult<f32> {
    let (a, b) = (F32Array::new(&cluster_a)?, F32Array::new(&cluster_b)?);
    let ((xa, na, da), (xb, nb, db)) = (rows(&a, "cluster_a")?, rows(&b, "cluster_b")?);
    if da != db {
        return Err(PyValueError::new_err("clusters must have the same number of features"));
    }
    Ok(d_prime(xa, na, xb, nb, da))
}

/// Isolation distance of a unit (Schmitzer-Torbert et al. 2005): the squared Mahalanobis distance,
/// from the unit's centre, of the n-th closest spike of other units (n = the unit's spike count).
/// Larger is better isolated.
///
/// Parameters
/// ----------
/// features : numpy.ndarray
///     `[spikes, features]` of all spikes.
/// labels : list of int
///     Unit of each spike.
/// target_unit : int
///     The unit scored.
///
/// Returns
/// -------
/// float
#[gen_stub_pyfunction]
#[pyfunction]
pub fn compute_isolation_distance(features: Bound<'_, PyAny>, labels: Vec<usize>, target_unit: usize) -> PyResult<f32> {
    let f = F32Array::new(&features)?;
    let (x, n, d) = rows(&f, "features")?;
    if n != labels.len() {
        return Err(PyValueError::new_err("one label per row of features"));
    }
    Ok(isolation_distance(x, &labels, d, target_unit))
}

/// Silhouette score of a unit in `[-1, 1]`: how much closer its spikes are to each other than to
/// the nearest other unit (1: well separated).
///
/// Parameters
/// ----------
/// features : numpy.ndarray
///     `[spikes, features]` of all spikes.
/// labels : list of int
///     Unit of each spike.
/// target_unit : int
///     The unit scored.
///
/// Returns
/// -------
/// float
#[gen_stub_pyfunction]
#[pyfunction]
pub fn compute_silhouette_score(features: Bound<'_, PyAny>, labels: Vec<usize>, target_unit: usize) -> PyResult<f32> {
    let f = F32Array::new(&features)?;
    let (x, n, d) = rows(&f, "features")?;
    if n != labels.len() {
        return Err(PyValueError::new_err("one label per row of features"));
    }
    Ok(silhouette(x, &labels, d, target_unit))
}

/// Amplitude cutoff (IBL / SpikeInterface): the estimated fraction of a unit's spikes missed because
/// they fall below the detection threshold, from the shape of its amplitude distribution.
///
/// Parameters
/// ----------
/// amplitudes : list of float
///     Spike amplitudes of the unit.
///
/// Returns
/// -------
/// float
///     In `[0, 0.5]`; low is good.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn compute_amplitude_cutoff(amplitudes: Vec<f32>) -> f64 {
    amplitude_cutoff(&amplitudes)
}

/// Presence ratio (SpikeInterface): the fraction of time bins in which the unit fires.
///
/// Parameters
/// ----------
/// spike_samples : list of int
///     Spike times, recording samples.
/// total_samples : int
///     Length of the recording, samples.
/// fs : float
///     Sampling rate, Hz.
/// bin_duration_sec : float, default 60.0
///     Bin width, s.
/// mean_fr_ratio : float, default 0.0
///     A bin counts when the unit fires above `mean_fr_ratio` × its mean rate there (0: any spike).
///
/// Returns
/// -------
/// float
///     In `[0, 1]`; 1: present throughout.
#[gen_stub_pyfunction]
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
