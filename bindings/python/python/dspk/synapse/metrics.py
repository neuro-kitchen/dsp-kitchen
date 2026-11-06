"""Unit quality and spike-train analyses. Spike trains are sample indices (any integer array);
``sample_rate`` is in Hz. Defaults follow SpikeInterface / the cited papers.
"""

from __future__ import annotations

import numpy as np

from dspk._native import synapse as _synapse

_m = _synapse.metrics

DEFAULT_ISI_THRESHOLD_MS: float = _m.DEFAULT_ISI_THRESHOLD_MS
DEFAULT_MIN_ISI_MS: float = _m.DEFAULT_MIN_ISI_MS
DEFAULT_PRESENCE_BIN_S: float = _m.DEFAULT_PRESENCE_BIN_S
DEFAULT_PRESENCE_MEAN_RATE_RATIO: float = _m.DEFAULT_PRESENCE_MEAN_RATE_RATIO
DEFAULT_CORRELOGRAM_BIN_MS: float = _m.DEFAULT_CORRELOGRAM_BIN_MS
DEFAULT_CORRELOGRAM_WINDOW_MS: float = _m.DEFAULT_CORRELOGRAM_WINDOW_MS
DEFAULT_MATCH_DELTA_MS: float = _m.DEFAULT_MATCH_DELTA_MS
DEFAULT_AGREEMENT_THRESHOLD: float = _m.DEFAULT_AGREEMENT_THRESHOLD
AMPLITUDE_CUTOFF_BINS: int = _m.AMPLITUDE_CUTOFF_BINS
AMPLITUDE_CUTOFF_SMOOTHING: float = _m.AMPLITUDE_CUTOFF_SMOOTHING
AMPLITUDE_CUTOFF_MIN_RATIO: float = _m.AMPLITUDE_CUTOFF_MIN_RATIO
#: Kilosort4's refractoriness test (``acg_refractory`` / ``ccg_refractory``).
DEFAULT_REFRACTORY_BIN_MS: float = _m.DEFAULT_REFRACTORY_BIN_MS
DEFAULT_REFRACTORY_WINDOW_MS: float = _m.DEFAULT_REFRACTORY_WINDOW_MS
DEFAULT_ACG_RATIO: float = _m.DEFAULT_ACG_RATIO
DEFAULT_ACG_PROBABILITY: float = _m.DEFAULT_ACG_PROBABILITY
DEFAULT_CCG_RATIO: float = _m.DEFAULT_CCG_RATIO
DEFAULT_CCG_PROBABILITY: float = _m.DEFAULT_CCG_PROBABILITY


def _train(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.uint64)


def _f64(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def isi_violations(train, sample_rate: float, duration_s: float, *, threshold_ms: float = DEFAULT_ISI_THRESHOLD_MS, min_isi_ms: float = DEFAULT_MIN_ISI_MS) -> dict:
    """ISI violations (Hill et al. 2011, SpikeInterface's ``isi_violations_ratio``): ``violations``,
    ``ratio``, ``rate_pct``, ``per_second``, ``firing_rate_hz``, ``spikes``."""
    return _m.isi_violations(_train(train), sample_rate, duration_s, threshold_ms, min_isi_ms)


def isi_histogram(train, sample_rate: float, *, bin_ms: float = 1.0, max_ms: float = 100.0) -> dict:
    """Inter-spike-interval histogram: ``bin_ms`` (left edges) and ``counts``."""
    return _m.isi_histogram(_train(train), sample_rate, bin_ms, max_ms)


def presence_ratio(train, total_samples: int, sample_rate: float, *, bin_s: float = DEFAULT_PRESENCE_BIN_S, mean_rate_ratio: float = DEFAULT_PRESENCE_MEAN_RATE_RATIO) -> float:
    """Fraction of ``bin_s`` bins where the unit fires above ``mean_rate_ratio`` × its mean rate."""
    return _m.presence_ratio(_train(train), int(total_samples), sample_rate, bin_s, mean_rate_ratio)


def amplitude_cutoff(amplitudes, *, bins: int = AMPLITUDE_CUTOFF_BINS, smoothing_bins: float = AMPLITUDE_CUTOFF_SMOOTHING, min_ratio: float = AMPLITUDE_CUTOFF_MIN_RATIO) -> float:
    """Estimated fraction of spikes below the detection threshold, in ``[0, 0.5]``."""
    return _m.amplitude_cutoff(np.ascontiguousarray(amplitudes, dtype=np.float32), bins, smoothing_bins, min_ratio)


def refractory_violations(train, refractory_samples: int) -> int:
    """Spike pairs at most ``refractory_samples`` apart (all pairs, not only consecutive)."""
    return _m.refractory_violations(_train(train), int(refractory_samples))


def llobet_contamination(train, total_samples: int, sample_rate: float, *, refractory_ms: float = DEFAULT_ISI_THRESHOLD_MS, censored_ms: float = DEFAULT_MIN_ISI_MS) -> float:
    """Refractory-period contamination (Llobet et al. 2022)."""
    return _m.llobet_contamination(_train(train), int(total_samples), sample_rate, refractory_ms, censored_ms)


def snr(peak_uv: float, noise_std_uv: float) -> float:
    """``|peak| / noise_std`` (NaN when the noise is not positive)."""
    return _m.snr(peak_uv, noise_std_uv)


def d_prime(a, b) -> float:
    """Linear-discriminant d′ between two clusters' features ``(n_a, d)`` and ``(n_b, d)``."""
    return _m.d_prime(np.ascontiguousarray(a, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32))


def silhouette(features, labels, unit: int) -> float:
    """Silhouette score of ``unit`` in ``[-1, 1]`` among labelled ``features`` ``(n, d)``."""
    return _m.silhouette(np.ascontiguousarray(features, dtype=np.float32), [int(x) for x in labels], unit)


def isolation_distance(features, labels, unit: int) -> float:
    """Isolation distance of ``unit`` (Schmitzer-Torbert et al. 2005)."""
    return _m.isolation_distance(np.ascontiguousarray(features, dtype=np.float32), [int(x) for x in labels], unit)


def autocorrelogram(train, sample_rate: float, *, bin_ms: float = DEFAULT_CORRELOGRAM_BIN_MS, window_ms: float = DEFAULT_CORRELOGRAM_WINDOW_MS) -> dict:
    """Auto-correlogram (zero lag excluded): ``lags_ms``, ``counts``."""
    return _m.autocorrelogram(_train(train), sample_rate, bin_ms, window_ms)


def crosscorrelogram(a, b, sample_rate: float, *, bin_ms: float = DEFAULT_CORRELOGRAM_BIN_MS, window_ms: float = DEFAULT_CORRELOGRAM_WINDOW_MS) -> dict:
    """Cross-correlogram of ``b`` around ``a`` (lags ``t_b − t_a``): ``lags_ms``, ``counts``."""
    return _m.crosscorrelogram(_train(a), _train(b), sample_rate, bin_ms, window_ms)


def acg_refractory(train, sample_rate: float, *, bin_ms: float = DEFAULT_REFRACTORY_BIN_MS, window_ms: float = DEFAULT_REFRACTORY_WINDOW_MS, ratio: float = DEFAULT_ACG_RATIO, probability: float = DEFAULT_ACG_PROBABILITY) -> dict:
    """Kilosort4's test of a refractory auto-correlogram: ``ratio``, ``probability``, ``baseline``, ``refractory``."""
    return _m.acg_refractory(_train(train), sample_rate, bin_ms, window_ms, ratio, probability)


def ccg_refractory(a, b, sample_rate: float, *, bin_ms: float = DEFAULT_REFRACTORY_BIN_MS, window_ms: float = DEFAULT_REFRACTORY_WINDOW_MS, ratio: float = DEFAULT_CCG_RATIO, probability: float = DEFAULT_CCG_PROBABILITY) -> dict:
    """Kilosort4's test of a refractory cross-correlogram (two trains likely one unit)."""
    return _m.ccg_refractory(_train(a), _train(b), sample_rate, bin_ms, window_ms, ratio, probability)


def firing_rate(spike_times_s, duration_s: float, *, bin_ms: float = 10.0, sigma_ms: float = 50.0) -> dict:
    """Gaussian-smoothed instantaneous firing rate: ``time_s``, ``rate_hz``."""
    return _m.firing_rate(_f64(spike_times_s), duration_s, bin_ms, sigma_ms)


def bursts(spike_times_s, duration_s: float, *, threshold_hz: float, min_duration_ms: float = 50.0, bin_ms: float = 10.0, sigma_ms: float = 50.0) -> dict:
    """Epochs where the smoothed rate stays above ``threshold_hz``: ``onset_s``, ``offset_s``, ``peak_hz``."""
    return _m.bursts(_f64(spike_times_s), duration_s, bin_ms, sigma_ms, threshold_hz, min_duration_ms)


def psth(spike_times_s, stimulus_times_s, *, pre_ms: float = 100.0, post_ms: float = 500.0, bin_ms: float = 10.0) -> dict:
    """Peri-stimulus time histogram: ``time_ms``, ``rate_hz`` (mean over trials), ``se_hz``, ``trials``."""
    return _m.psth(_f64(spike_times_s), _f64(stimulus_times_s), pre_ms, post_ms, bin_ms)


def stimulus_average(data, stimulus_samples, sample_rate: float, *, pre_ms: float = 10.0, post_ms: float = 50.0, response_ms: tuple[float, float] = (2.0, 40.0), threshold_sigma: float = 3.0) -> dict:
    """Stimulus-triggered average of ``data`` ``(channels, samples)``: ``time_ms``, ``mean`` /
    ``std`` / ``se`` ``(channels, samples)``, ``trials``; per channel the evoked potential in
    ``response_ms`` (``mep_onset_ms``, ``mep_peak_to_peak_uv``, ``mep_rms_uv``, ``mep_area_uv_ms``)."""
    pre, post = round(pre_ms * sample_rate / 1e3), round(post_ms * sample_rate / 1e3)
    return _m.stimulus_average(np.ascontiguousarray(data, dtype=np.float32), _train(stimulus_samples), pre, post, sample_rate, response_ms[0], response_ms[1], threshold_sigma)


def composite_score(train, amplitudes, snr: float, total_samples: int, sample_rate: float) -> dict:
    """EMUsort's composite quality score of a unit: ``score`` in ``[0, 1]`` and its parts."""
    return _m.composite_score(_train(train), np.ascontiguousarray(amplitudes, dtype=np.float32), snr, int(total_samples), sample_rate)


def firing_range(train, total_samples: int, sample_rate: float, *, bin_s: float = 0.5, percentiles: tuple[float, float] = (5.0, 95.0)) -> float:
    """Spread (Hz) of a unit's firing rate between two percentiles of ``bin_s`` bins."""
    return _m.firing_range(_train(train), int(total_samples), sample_rate, bin_s, percentiles[0], percentiles[1])


def compare_trains(a, b, *, tolerance_samples: int) -> dict:
    """Two trains matched 1-to-1 within ``tolerance_samples``: ``matches``, ``false_negatives``,
    ``false_positives``, ``agreement``, ``precision``, ``recall``."""
    return _m.compare_trains(_train(a), _train(b), int(tolerance_samples))


def compare_sortings(a, b, *, delta_ms: float = DEFAULT_MATCH_DELTA_MS, threshold: float = DEFAULT_AGREEMENT_THRESHOLD) -> dict:
    """Two :class:`~dspk.synapse.io.Sorting` compared unit by unit: ``agreement`` ``(units_a, units_b)``,
    ``unit_ids_a``, ``unit_ids_b``, ``matched`` ``(a, b, agreement)``, ``well_detected`` (units of ``a``)."""
    return _m.compare_sortings(a._native, b._native, delta_ms, threshold)


__all__ = [
    "AMPLITUDE_CUTOFF_BINS", "AMPLITUDE_CUTOFF_MIN_RATIO", "AMPLITUDE_CUTOFF_SMOOTHING", "DEFAULT_ACG_PROBABILITY", "DEFAULT_ACG_RATIO",
    "DEFAULT_AGREEMENT_THRESHOLD", "DEFAULT_CCG_PROBABILITY", "DEFAULT_CCG_RATIO", "DEFAULT_CORRELOGRAM_BIN_MS", "DEFAULT_CORRELOGRAM_WINDOW_MS",
    "DEFAULT_ISI_THRESHOLD_MS", "DEFAULT_MATCH_DELTA_MS", "DEFAULT_MIN_ISI_MS", "DEFAULT_PRESENCE_BIN_S", "DEFAULT_PRESENCE_MEAN_RATE_RATIO",
    "DEFAULT_REFRACTORY_BIN_MS", "DEFAULT_REFRACTORY_WINDOW_MS",
    "acg_refractory", "amplitude_cutoff", "autocorrelogram", "bursts", "ccg_refractory", "compare_sortings", "compare_trains", "composite_score",
    "crosscorrelogram", "d_prime", "firing_range", "firing_rate", "isi_histogram", "isi_violations", "isolation_distance", "llobet_contamination",
    "presence_ratio", "psth", "refractory_violations", "silhouette", "snr", "stimulus_average",
]
