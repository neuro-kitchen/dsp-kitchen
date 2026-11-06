//! `dspk.synapse.metrics`: quality of units (firing, refractoriness, isolation), correlograms,
//! firing rate and bursts, evoked responses, and sorting comparison. Spike trains are sample
//! indices (`uint64`); results are numbers or dicts of NumPy arrays.

use dspk_synapse::metrics::{self as m, CompositeScoreOptions, Correlogram, RefractoryOptions, RefractoryStats};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::io::Sorting;

type Dict<'py> = Bound<'py, PyDict>;

fn invalid(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn sorted(train: &[u64]) -> Vec<u64> {
    let mut t = train.to_vec();
    t.sort_unstable();
    t
}

fn correlogram<'py>(py: Python<'py>, c: Correlogram) -> PyResult<Dict<'py>> {
    let out = PyDict::new(py);
    out.set_item("lags_ms", c.bin_centers_ms.into_pyarray(py))?;
    out.set_item("counts", c.counts.into_pyarray(py))?;
    out.set_item("bin_ms", c.bin_size_ms)?;
    out.set_item("window_ms", c.window_ms)?;
    Ok(out)
}

fn refractory<'py>(py: Python<'py>, (s, refractory): (RefractoryStats, bool)) -> PyResult<Dict<'py>> {
    let out = PyDict::new(py);
    out.set_item("ratio", s.ratio)?;
    out.set_item("probability", s.probability)?;
    out.set_item("baseline", s.baseline)?;
    out.set_item("refractory", refractory)?;
    Ok(out)
}

/// ISI violations (Hill et al. 2011, as SpikeInterface): `violations`, `ratio`
/// (`isi_violations_ratio`), `rate_pct`, `per_second`, `firing_rate_hz`, `spikes`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn isi_violations<'py>(py: Python<'py>, train: PyReadonlyArray1<'_, u64>, sample_rate: f64, duration_s: f64, threshold_ms: f64, min_isi_ms: f64) -> PyResult<Dict<'py>> {
    let r = m::compute_isi_violations(train.as_slice()?, sample_rate, duration_s, threshold_ms, min_isi_ms);
    let out = PyDict::new(py);
    out.set_item("violations", r.violation_count)?;
    out.set_item("ratio", r.isi_violations_ratio)?;
    out.set_item("rate_pct", r.violation_rate_pct)?;
    out.set_item("per_second", r.violations_per_sec)?;
    out.set_item("firing_rate_hz", r.firing_rate_hz)?;
    out.set_item("spikes", r.total_spikes)?;
    Ok(out)
}

/// Inter-spike-interval histogram: `(bin_edges_ms, counts)` over `[0, max_ms]` in `bin_ms` bins.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn isi_histogram<'py>(py: Python<'py>, train: PyReadonlyArray1<'_, u64>, sample_rate: f64, bin_ms: f64, max_ms: f64) -> PyResult<Dict<'py>> {
    let (edges, counts) = m::isi_histogram(train.as_slice()?, sample_rate, bin_ms, max_ms);
    let out = PyDict::new(py);
    out.set_item("bin_ms", edges.into_pyarray(py))?;
    out.set_item("counts", counts.into_pyarray(py))?;
    Ok(out)
}

/// Fraction of `bin_s` bins where the unit fires more than `mean_rate_ratio` × its mean rate.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn presence_ratio(train: PyReadonlyArray1<'_, u64>, total_samples: u64, sample_rate: f64, bin_s: f64, mean_rate_ratio: f64) -> PyResult<f64> {
    Ok(m::compute_presence_ratio(train.as_slice()?, total_samples, sample_rate, bin_s, mean_rate_ratio))
}

/// Amplitude cutoff in `[0, 0.5]`: the estimated fraction of spikes below the detection threshold
/// (NaN with too few spikes).
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn amplitude_cutoff(amplitudes: PyReadonlyArray1<'_, f32>, bins: usize, smoothing_bins: f64, min_ratio: f64) -> PyResult<f64> {
    Ok(m::compute_amplitude_cutoff_with(amplitudes.as_slice()?, bins, smoothing_bins, min_ratio))
}

/// Spike pairs (not only consecutive ones) at most `refractory_samples` apart.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn refractory_violations(train: PyReadonlyArray1<'_, u64>, refractory_samples: u64) -> PyResult<u64> {
    Ok(m::count_refractory_violations(&sorted(train.as_slice()?), refractory_samples))
}

/// Refractory-period contamination (Llobet et al. 2022).
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn llobet_contamination(train: PyReadonlyArray1<'_, u64>, total_samples: u64, sample_rate: f64, refractory_ms: f64, censored_ms: f64) -> PyResult<f64> {
    Ok(m::compute_llobet_contamination(train.as_slice()?, total_samples, sample_rate, refractory_ms, censored_ms))
}

/// Signal-to-noise ratio `|peak| / noise_std` (NaN when the noise is not positive).
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn snr(peak_uv: f32, noise_std_uv: f32) -> f32 {
    m::compute_snr(peak_uv, noise_std_uv)
}

fn features(f: &PyReadonlyArray2<'_, f32>) -> PyResult<(Vec<f32>, usize, usize)> {
    let a = f.as_array();
    Ok((a.iter().copied().collect(), a.nrows(), a.ncols()))
}

/// Linear-discriminant d′ between two clusters' features (`(n_a, d)`, `(n_b, d)`).
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn d_prime(a: PyReadonlyArray2<'_, f32>, b: PyReadonlyArray2<'_, f32>) -> PyResult<f32> {
    let ((fa, na, da), (fb, nb, db)) = (features(&a)?, features(&b)?);
    if da != db {
        return Err(invalid(format!("feature dimensions differ: {da} vs {db}")));
    }
    Ok(m::compute_d_prime(&fa, na, &fb, nb, da))
}

/// Silhouette score of `unit` in `[-1, 1]` among labelled features `(n, d)`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn silhouette(features_: PyReadonlyArray2<'_, f32>, labels: Vec<usize>, unit: usize) -> PyResult<f32> {
    let (f, n, d) = features(&features_)?;
    if labels.len() != n {
        return Err(invalid(format!("{} labels for {n} rows", labels.len())));
    }
    Ok(m::compute_silhouette_score(&f, &labels, d, unit))
}

/// Isolation distance of `unit` (Schmitzer-Torbert et al. 2005) among labelled features `(n, d)`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn isolation_distance(features_: PyReadonlyArray2<'_, f32>, labels: Vec<usize>, unit: usize) -> PyResult<f32> {
    let (f, n, d) = features(&features_)?;
    if labels.len() != n {
        return Err(invalid(format!("{} labels for {n} rows", labels.len())));
    }
    Ok(m::compute_isolation_distance(&f, &labels, d, unit))
}

/// Auto-correlogram (zero lag excluded): `lags_ms`, `counts`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn autocorrelogram<'py>(py: Python<'py>, train: PyReadonlyArray1<'_, u64>, sample_rate: f64, bin_ms: f32, window_ms: f32) -> PyResult<Dict<'py>> {
    correlogram(py, m::compute_autocorrelogram(&sorted(train.as_slice()?), sample_rate, bin_ms, window_ms))
}

/// Cross-correlogram of `b` around `a` (lags `t_b − t_a`): `lags_ms`, `counts`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn crosscorrelogram<'py>(py: Python<'py>, a: PyReadonlyArray1<'_, u64>, b: PyReadonlyArray1<'_, u64>, sample_rate: f64, bin_ms: f32, window_ms: f32) -> PyResult<Dict<'py>> {
    correlogram(py, m::compute_crosscorrelogram(&sorted(a.as_slice()?), &sorted(b.as_slice()?), sample_rate, bin_ms, window_ms))
}

fn refractory_options(bin_ms: f32, window_ms: f32, ratio: f64, probability: f64) -> RefractoryOptions {
    RefractoryOptions { bin_ms, window_ms, acg_ratio: ratio, acg_probability: probability, ccg_ratio: ratio, ccg_probability: probability, ..Default::default() }
}

/// Whether a unit's auto-correlogram looks refractory (Kilosort4's test): `ratio`,
/// `probability`, `baseline`, `refractory`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn acg_refractory<'py>(py: Python<'py>, train: PyReadonlyArray1<'_, u64>, sample_rate: f64, bin_ms: f32, window_ms: f32, ratio: f64, probability: f64) -> PyResult<Dict<'py>> {
    refractory(py, m::acg_refractory(&sorted(train.as_slice()?), sample_rate, &refractory_options(bin_ms, window_ms, ratio, probability)))
}

/// Whether two units' cross-correlogram looks refractory (they are likely one unit): as
/// [`acg_refractory`].
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn ccg_refractory<'py>(py: Python<'py>, a: PyReadonlyArray1<'_, u64>, b: PyReadonlyArray1<'_, u64>, sample_rate: f64, bin_ms: f32, window_ms: f32, ratio: f64, probability: f64) -> PyResult<Dict<'py>> {
    refractory(py, m::ccg_refractory(&sorted(a.as_slice()?), &sorted(b.as_slice()?), sample_rate, &refractory_options(bin_ms, window_ms, ratio, probability)))
}

/// Gaussian-smoothed instantaneous firing rate: `time_s`, `rate_hz`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn firing_rate<'py>(py: Python<'py>, spike_times_s: PyReadonlyArray1<'_, f64>, duration_s: f64, bin_ms: f64, sigma_ms: f64) -> PyResult<Dict<'py>> {
    let c = m::compute_instantaneous_firing_rate(spike_times_s.as_slice()?, duration_s, bin_ms, sigma_ms);
    let out = PyDict::new(py);
    out.set_item("time_s", c.time_bin_centers_sec.into_pyarray(py))?;
    out.set_item("rate_hz", c.rate_hz.into_pyarray(py))?;
    Ok(out)
}

/// Burst epochs where the smoothed firing rate stays above `threshold_hz` for `min_duration_ms`:
/// `onset_s`, `offset_s`, `peak_hz` arrays.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn bursts<'py>(py: Python<'py>, spike_times_s: PyReadonlyArray1<'_, f64>, duration_s: f64, bin_ms: f64, sigma_ms: f64, threshold_hz: f32, min_duration_ms: f64) -> PyResult<Dict<'py>> {
    let curve = m::compute_instantaneous_firing_rate(spike_times_s.as_slice()?, duration_s, bin_ms, sigma_ms);
    let epochs = m::detect_burst_epochs(&curve, threshold_hz, min_duration_ms);
    let out = PyDict::new(py);
    out.set_item("onset_s", epochs.iter().map(|e| e.onset_sec).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("offset_s", epochs.iter().map(|e| e.offset_sec).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("peak_hz", epochs.iter().map(|e| e.peak_rate_hz).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(out)
}

/// Peri-stimulus time histogram: `time_ms`, `rate_hz` (mean over trials), `se_hz`, `trials`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn psth<'py>(py: Python<'py>, spike_times_s: PyReadonlyArray1<'_, f64>, stimulus_times_s: PyReadonlyArray1<'_, f64>, pre_ms: f64, post_ms: f64, bin_ms: f64) -> PyResult<Dict<'py>> {
    let r = m::compute_psth(spike_times_s.as_slice()?, stimulus_times_s.as_slice()?, pre_ms, post_ms, bin_ms);
    let out = PyDict::new(py);
    out.set_item("time_ms", r.time_bins_ms.into_pyarray(py))?;
    out.set_item("rate_hz", r.mean_rate_hz.into_pyarray(py))?;
    out.set_item("se_hz", r.se_rate_hz.into_pyarray(py))?;
    out.set_item("trials", r.num_trials)?;
    Ok(out)
}

/// Stimulus-triggered average of `data` `(channels, samples)` around `stimulus_samples`:
/// `time_ms`, `mean`, `std`, `se` `(channels, pre + post)`, `trials`; and per channel the motor
/// evoked potential (`mep_*`: onset latency, peak-to-peak, RMS, rectified area) in
/// `[response_start_ms, response_end_ms]`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn stimulus_average<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'_, f32>,
    stimulus_samples: PyReadonlyArray1<'_, u64>,
    pre_samples: usize,
    post_samples: usize,
    sample_rate: f64,
    response_start_ms: f64,
    response_end_ms: f64,
    threshold_sigma: f32,
) -> PyResult<Dict<'py>> {
    let a = data.as_array();
    let (channels, samples) = (a.nrows(), a.ncols());
    let flat: Vec<f32> = a.iter().copied().collect();
    let sta = m::compute_stimulus_triggered_average(&flat, channels, samples, stimulus_samples.as_slice()?, pre_samples, post_samples, sample_rate);
    let meps = m::quantify_mep(&sta, response_start_ms, response_end_ms, threshold_sigma);
    let shape = (sta.num_channels, sta.num_samples);
    let out = PyDict::new(py);
    out.set_item("time_ms", sta.time_ms.clone().into_pyarray(py))?;
    for (key, v) in [("mean", &sta.mean_uv), ("std", &sta.std_uv), ("se", &sta.se_uv)] {
        out.set_item(key, Array2::from_shape_vec(shape, v.clone()).map_err(invalid)?.into_pyarray(py))?;
    }
    out.set_item("trials", sta.num_trials)?;
    out.set_item("mep_onset_ms", meps.iter().map(|x| x.onset_latency_ms).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("mep_peak_to_peak_uv", meps.iter().map(|x| x.peak_to_peak_uv).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("mep_rms_uv", meps.iter().map(|x| x.rms_uv).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("mep_area_uv_ms", meps.iter().map(|x| x.rectified_auc_uv_ms).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(out)
}

/// EMUsort's composite score of one unit: `score` in `[0, 1]` and its parts `type_1`, `type_2`,
/// `firing_rate`, `snr`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn composite_score<'py>(py: Python<'py>, train: PyReadonlyArray1<'_, u64>, amplitudes: PyReadonlyArray1<'_, f32>, snr: f64, total_samples: u64, sample_rate: f64) -> PyResult<Dict<'py>> {
    let s = m::composite_score(&sorted(train.as_slice()?), amplitudes.as_slice()?, snr, total_samples, sample_rate, &CompositeScoreOptions::default());
    let out = PyDict::new(py);
    out.set_item("score", s.score)?;
    out.set_item("type_1", s.type_1)?;
    out.set_item("type_2", s.type_2)?;
    out.set_item("firing_rate", s.firing_rate)?;
    out.set_item("snr", s.snr)?;
    Ok(out)
}

/// Spread of a unit's firing rate (Hz) between the `low` and `high` percentiles of `bin_s` bins.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn firing_range(train: PyReadonlyArray1<'_, u64>, total_samples: u64, sample_rate: f64, bin_s: f64, low: f64, high: f64) -> PyResult<f64> {
    Ok(m::firing_range(train.as_slice()?, total_samples, sample_rate, bin_s, (low, high)))
}

/// Two spike trains matched 1-to-1 within `tolerance_samples`: `matches`, `false_negatives`,
/// `false_positives`, `agreement`, `precision`, `recall`.
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn compare_trains<'py>(py: Python<'py>, a: PyReadonlyArray1<'_, u64>, b: PyReadonlyArray1<'_, u64>, tolerance_samples: u64) -> PyResult<Dict<'py>> {
    let r = m::compare_spike_trains(&sorted(a.as_slice()?), &sorted(b.as_slice()?), tolerance_samples);
    let out = PyDict::new(py);
    out.set_item("matches", r.num_matches)?;
    out.set_item("false_negatives", r.false_negatives)?;
    out.set_item("false_positives", r.false_positives)?;
    out.set_item("agreement", r.agreement_score)?;
    out.set_item("precision", r.precision)?;
    out.set_item("recall", r.recall)?;
    Ok(out)
}

/// Two sortings compared unit by unit (spikes matched within `delta_ms`): `agreement`
/// `(units_a, units_b)`, `unit_ids_a`, `unit_ids_b`, the `matched` pairs `(a, b, agreement)`, and
/// the units of `a` found with at least `threshold` agreement (`well_detected`).
#[gen_stub_pyfunction(module = "dspk._native.synapse.metrics")]
#[pyfunction]
pub fn compare_sortings<'py>(py: Python<'py>, a: &Sorting, b: &Sorting, delta_ms: f64, threshold: f32) -> PyResult<Dict<'py>> {
    let c = py.detach(|| m::compare_sortings(&a.inner, &b.inner, delta_ms, threshold));
    let out = PyDict::new(py);
    let shape = (c.unit_ids_a.len(), c.unit_ids_b.len());
    out.set_item("agreement", Array2::from_shape_vec(shape, c.agreement_matrix.clone()).map_err(invalid)?.into_pyarray(py))?;
    out.set_item("unit_ids_a", c.unit_ids_a.clone())?;
    out.set_item("unit_ids_b", c.unit_ids_b.clone())?;
    out.set_item("matched", c.matched_units.iter().map(|u| (u.unit_id_a, u.unit_id_b, u.metrics.agreement_score)).collect::<Vec<_>>())?;
    out.set_item("well_detected", c.well_detected_units_a.clone())?;
    Ok(out)
}

pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_ISI_THRESHOLD_MS", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_MIN_ISI_MS", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_PRESENCE_BIN_S", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_PRESENCE_MEAN_RATE_RATIO", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_CORRELOGRAM_BIN_MS", f32);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_CORRELOGRAM_WINDOW_MS", f32);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_MATCH_DELTA_MS", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_AGREEMENT_THRESHOLD", f32);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "AMPLITUDE_CUTOFF_BINS", usize);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "AMPLITUDE_CUTOFF_SMOOTHING", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "AMPLITUDE_CUTOFF_MIN_RATIO", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_REFRACTORY_BIN_MS", f32);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_REFRACTORY_WINDOW_MS", f32);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_ACG_RATIO", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_ACG_PROBABILITY", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_CCG_RATIO", f64);
pyo3_stub_gen::module_variable!("dspk._native.synapse.metrics", "DEFAULT_CCG_PROBABILITY", f64);

pub(crate) fn register(md: &Bound<'_, PyModule>) -> PyResult<()> {
    md.add("DEFAULT_ISI_THRESHOLD_MS", m::DEFAULT_ISI_THRESHOLD_MS)?;
    md.add("DEFAULT_MIN_ISI_MS", m::DEFAULT_MIN_ISI_MS)?;
    md.add("DEFAULT_PRESENCE_BIN_S", m::DEFAULT_PRESENCE_BIN_SEC)?;
    md.add("DEFAULT_PRESENCE_MEAN_RATE_RATIO", m::DEFAULT_PRESENCE_MEAN_FR_RATIO)?;
    md.add("DEFAULT_CORRELOGRAM_BIN_MS", m::DEFAULT_CORRELOGRAM_BIN_MS)?;
    md.add("DEFAULT_CORRELOGRAM_WINDOW_MS", m::DEFAULT_CORRELOGRAM_WINDOW_MS)?;
    md.add("DEFAULT_MATCH_DELTA_MS", m::DEFAULT_MATCH_DELTA_MS)?;
    md.add("DEFAULT_AGREEMENT_THRESHOLD", m::DEFAULT_AGREEMENT_THRESHOLD)?;
    md.add("AMPLITUDE_CUTOFF_BINS", m::firing::AMPLITUDE_CUTOFF_BINS)?;
    md.add("AMPLITUDE_CUTOFF_SMOOTHING", m::firing::AMPLITUDE_CUTOFF_SMOOTHING)?;
    md.add("AMPLITUDE_CUTOFF_MIN_RATIO", m::firing::AMPLITUDE_CUTOFF_MIN_RATIO)?;
    let r = RefractoryOptions::default();
    md.add("DEFAULT_REFRACTORY_BIN_MS", r.bin_ms)?;
    md.add("DEFAULT_REFRACTORY_WINDOW_MS", r.window_ms)?;
    md.add("DEFAULT_ACG_RATIO", r.acg_ratio)?;
    md.add("DEFAULT_ACG_PROBABILITY", r.acg_probability)?;
    md.add("DEFAULT_CCG_RATIO", r.ccg_ratio)?;
    md.add("DEFAULT_CCG_PROBABILITY", r.ccg_probability)?;
    for f in [
        wrap_pyfunction!(isi_violations, md)?,
        wrap_pyfunction!(isi_histogram, md)?,
        wrap_pyfunction!(presence_ratio, md)?,
        wrap_pyfunction!(amplitude_cutoff, md)?,
        wrap_pyfunction!(refractory_violations, md)?,
        wrap_pyfunction!(llobet_contamination, md)?,
        wrap_pyfunction!(snr, md)?,
        wrap_pyfunction!(d_prime, md)?,
        wrap_pyfunction!(silhouette, md)?,
        wrap_pyfunction!(isolation_distance, md)?,
        wrap_pyfunction!(autocorrelogram, md)?,
        wrap_pyfunction!(crosscorrelogram, md)?,
        wrap_pyfunction!(acg_refractory, md)?,
        wrap_pyfunction!(ccg_refractory, md)?,
        wrap_pyfunction!(firing_rate, md)?,
        wrap_pyfunction!(bursts, md)?,
        wrap_pyfunction!(psth, md)?,
        wrap_pyfunction!(stimulus_average, md)?,
        wrap_pyfunction!(composite_score, md)?,
        wrap_pyfunction!(firing_range, md)?,
        wrap_pyfunction!(compare_trains, md)?,
        wrap_pyfunction!(compare_sortings, md)?,
    ] {
        md.add_function(f)?;
    }
    Ok(())
}
