"""Unit quality and spike-train analyses. Spike trains are sample indices (any integer array);
``sample_rate`` is in Hz. Defaults follow SpikeInterface / the cited papers.
"""

from __future__ import annotations

import numpy as np

from dspk._native import synapse as _synapse

_m = _synapse.metrics

DEFAULT_ISI_THRESHOLD_MS: float = _m.DEFAULT_ISI_THRESHOLD_MS
DEFAULT_PSTH_PRE_MS: float = _m.DEFAULT_PSTH_PRE_MS
DEFAULT_PSTH_POST_MS: float = _m.DEFAULT_PSTH_POST_MS
DEFAULT_PSTH_BIN_MS: float = _m.DEFAULT_PSTH_BIN_MS
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



def isi_violations(train, total_samples: int, sample_rate: float, *, threshold_ms: float = DEFAULT_ISI_THRESHOLD_MS, min_isi_ms: float = DEFAULT_MIN_ISI_MS) -> dict:
    """Inter-spike-interval violations (Hill et al., 2011), as SpikeInterface's
    ``isi_violations_ratio``.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    total_samples : int
        Samples in the recording.
    sample_rate : float
        Sample rate in Hz.
    threshold_ms : float
        Intervals shorter than this violate the refractory period.
    min_isi_ms : float
        Intervals shorter than this cannot occur (censored by detection).

    Returns
    -------
    dict
        ``violations``, ``ratio``, ``rate_pct``, ``per_second``, ``firing_rate_hz``, ``spikes``.

    Examples
    --------
    >>> import dspk
    >>> out = dspk.synapse.metrics.isi_violations([0, 10, 3_000, 6_000], 30_000, 30_000.0)
    >>> out["violations"]
    1
    """
    return _m.isi_violations(_train(train), int(total_samples), sample_rate, threshold_ms, min_isi_ms)


def isi_histogram(train, sample_rate: float, *, bin_ms: float = 1.0, max_ms: float = 100.0) -> dict:
    """Histogram of inter-spike intervals.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    sample_rate : float
        Sample rate in Hz.
    bin_ms : float
        Bin width.
    max_ms : float
        Longest interval counted.

    Returns
    -------
    dict of numpy.ndarray
        ``bin_ms`` (left edges) and ``counts``.
    """
    return _m.isi_histogram(_train(train), sample_rate, bin_ms, max_ms)


def presence_ratio(train, total_samples: int, sample_rate: float, *, bin_s: float = DEFAULT_PRESENCE_BIN_S, mean_rate_ratio: float = DEFAULT_PRESENCE_MEAN_RATE_RATIO) -> float:
    """Fraction of the recording in which the unit fires.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    total_samples : int
        Samples in the recording.
    sample_rate : float
        Sample rate in Hz.
    bin_s : float
        Bin length.
    mean_rate_ratio : float
        A bin counts when the unit fires above this times its mean rate there (0: at all).

    Returns
    -------
    float
        In ``[0, 1]``.
    """
    return _m.presence_ratio(_train(train), int(total_samples), sample_rate, bin_s, mean_rate_ratio)


def amplitude_cutoff(amplitudes, *, bins: int = AMPLITUDE_CUTOFF_BINS, smoothing_bins: float = AMPLITUDE_CUTOFF_SMOOTHING, min_ratio: float = AMPLITUDE_CUTOFF_MIN_RATIO) -> float:
    """Estimated fraction of a unit's spikes missed below the detection threshold.

    From the histogram of its amplitudes, assuming it is symmetric around its peak.

    Parameters
    ----------
    amplitudes : array_like
        The unit's spike amplitudes.
    bins : int
        Histogram bins.
    smoothing_bins : float
        Gaussian smoothing of the histogram, in bins.
    min_ratio : float
        Fewer spikes than this per bin: the estimate is not made (NaN).

    Returns
    -------
    float
        In ``[0, 0.5]``.
    """
    return _m.amplitude_cutoff(np.ascontiguousarray(amplitudes, dtype=np.float32), bins, smoothing_bins, min_ratio)


def refractory_violations(train, refractory_samples: int) -> int:
    """Pairs of spikes at most ``refractory_samples`` apart (all pairs, not only consecutive ones).

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    refractory_samples : int
        The refractory period, in samples.

    Returns
    -------
    int
    """
    return _m.refractory_violations(_train(train), int(refractory_samples))


def llobet_contamination(train, total_samples: int, sample_rate: float, *, refractory_ms: float = DEFAULT_ISI_THRESHOLD_MS, censored_ms: float = DEFAULT_MIN_ISI_MS) -> float:
    """Refractory-period contamination (Llobet et al., 2022): the estimated fraction of spikes from
    other neurons.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    total_samples : int
        Samples in the recording.
    sample_rate : float
        Sample rate in Hz.
    refractory_ms : float
        The refractory period.
    censored_ms : float
        Intervals shorter than this cannot occur (censored by detection).

    Returns
    -------
    float
    """
    return _m.llobet_contamination(_train(train), int(total_samples), sample_rate, refractory_ms, censored_ms)


def snr(peak_uv: float, noise_std_uv: float) -> float:
    """Signal-to-noise ratio: ``|peak| / noise_std``.

    Parameters
    ----------
    peak_uv : float
        The template's peak.
    noise_std_uv : float
        The channel's noise.

    Returns
    -------
    float
        NaN when the noise is not positive.
    """
    return _m.snr(peak_uv, noise_std_uv)


def d_prime(a, b) -> float:
    """Linear-discriminant d′ between two clusters: how well a line separates them.

    Parameters
    ----------
    a : array_like
        Features of one cluster ``(n_a, d)``.
    b : array_like
        Features of the other ``(n_b, d)``.

    Returns
    -------
    float
    """
    return _m.d_prime(np.ascontiguousarray(a, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32))


def silhouette(features, labels, unit: int) -> float:
    """Silhouette score of one unit among labelled features.

    Parameters
    ----------
    features : array_like
        ``(n, d)``.
    labels : array_like
        Unit of every point.
    unit : int
        The unit scored.

    Returns
    -------
    float
        In ``[-1, 1]``; higher is better separated.
    """
    return _m.silhouette(np.ascontiguousarray(features, dtype=np.float32), [int(x) for x in labels], unit)


def isolation_distance(features, labels, unit: int) -> float:
    """Isolation distance of one unit (Schmitzer-Torbert et al., 2005).

    Parameters
    ----------
    features : array_like
        ``(n, d)``.
    labels : array_like
        Unit of every point.
    unit : int
        The unit scored.

    Returns
    -------
    float
        Higher is better isolated.
    """
    return _m.isolation_distance(np.ascontiguousarray(features, dtype=np.float32), [int(x) for x in labels], unit)


def autocorrelogram(train, sample_rate: float, *, bin_ms: float = DEFAULT_CORRELOGRAM_BIN_MS, window_ms: float = DEFAULT_CORRELOGRAM_WINDOW_MS) -> dict:
    """Auto-correlogram of a spike train, zero lag excluded.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    sample_rate : float
        Sample rate in Hz.
    bin_ms : float
        Bin width.
    window_ms : float
        Lags counted, ``-window_ms`` to ``window_ms``.

    Returns
    -------
    dict of numpy.ndarray
        ``lags_ms`` and ``counts``.
    """
    return _m.autocorrelogram(_train(train), sample_rate, bin_ms, window_ms)


def crosscorrelogram(a, b, sample_rate: float, *, bin_ms: float = DEFAULT_CORRELOGRAM_BIN_MS, window_ms: float = DEFAULT_CORRELOGRAM_WINDOW_MS) -> dict:
    """Cross-correlogram of ``b`` around ``a``: lags ``t_b - t_a``.

    Parameters
    ----------
    a, b : array_like
        Spike times, as sample indices.
    sample_rate : float
        Sample rate in Hz.
    bin_ms : float
        Bin width.
    window_ms : float
        Lags counted, ``-window_ms`` to ``window_ms``.

    Returns
    -------
    dict of numpy.ndarray
        ``lags_ms`` and ``counts``.
    """
    return _m.crosscorrelogram(_train(a), _train(b), sample_rate, bin_ms, window_ms)


def acg_refractory(train, sample_rate: float, *, bin_ms: float = DEFAULT_REFRACTORY_BIN_MS, window_ms: float = DEFAULT_REFRACTORY_WINDOW_MS, ratio: float = DEFAULT_ACG_RATIO, probability: float = DEFAULT_ACG_PROBABILITY) -> dict:
    """Kilosort4's test of whether a unit's auto-correlogram shows a refractory period.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    sample_rate : float
        Sample rate in Hz.
    bin_ms : float
        Bin width.
    window_ms : float
        Lags counted.
    ratio : float
        Refractory when the dip's ratio to the baseline (``R``) is below this...
    probability : float
        ...and its probability under a Poisson model (``Q``) is below this.

    Returns
    -------
    dict
        ``ratio``, ``probability``, ``baseline``, ``refractory`` (the verdict).
    """
    return _m.acg_refractory(_train(train), sample_rate, bin_ms, window_ms, ratio, probability)


def ccg_refractory(a, b, sample_rate: float, *, bin_ms: float = DEFAULT_REFRACTORY_BIN_MS, window_ms: float = DEFAULT_REFRACTORY_WINDOW_MS, ratio: float = DEFAULT_CCG_RATIO, probability: float = DEFAULT_CCG_PROBABILITY) -> dict:
    """Kilosort4's test of a refractory cross-correlogram: two trains that are likely one unit.

    Parameters
    ----------
    a, b : array_like
        Spike times, as sample indices.
    sample_rate : float
        Sample rate in Hz.
    bin_ms : float
        Bin width.
    window_ms : float
        Lags counted.
    ratio : float
        Refractory when the dip's ratio to the baseline (``R``) is below this...
    probability : float
        ...and its probability under a Poisson model (``Q``) is below this.

    Returns
    -------
    dict
        ``ratio``, ``probability``, ``baseline``, ``refractory`` (the verdict).
    """
    return _m.ccg_refractory(_train(a), _train(b), sample_rate, bin_ms, window_ms, ratio, probability)


def firing_rate(train, total_samples: int, sample_rate: float, *, bin_ms: float = 10.0, sigma_ms: float = 50.0) -> dict:
    """Instantaneous firing rate over the recording, smoothed with a Gaussian.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    total_samples : int
        Samples in the recording.
    sample_rate : float
        Sample rate in Hz.
    bin_ms : float
        Bin width.
    sigma_ms : float
        Standard deviation of the smoothing.

    Returns
    -------
    dict of numpy.ndarray
        ``time_s`` and ``rate_hz``.
    """
    return _m.firing_rate(_train(train), int(total_samples), sample_rate, bin_ms, sigma_ms)


def bursts(train, total_samples: int, sample_rate: float, *, threshold_hz: float, min_duration_ms: float = 50.0, bin_ms: float = 10.0, sigma_ms: float = 50.0) -> dict:
    """Epochs where the smoothed firing rate stays above a threshold.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    total_samples : int
        Samples in the recording.
    sample_rate : float
        Sample rate in Hz.
    threshold_hz : float
        Rate a burst exceeds.
    min_duration_ms : float
        Shorter epochs are not bursts.
    bin_ms : float
        Bin width of the rate.
    sigma_ms : float
        Smoothing of the rate.

    Returns
    -------
    dict of numpy.ndarray
        ``onset_s``, ``offset_s`` and ``peak_hz`` of each burst.
    """
    return _m.bursts(_train(train), int(total_samples), sample_rate, bin_ms, sigma_ms, threshold_hz, min_duration_ms)


def psth(train, stimulus_samples, sample_rate: float, *, pre_ms: float = DEFAULT_PSTH_PRE_MS, post_ms: float = DEFAULT_PSTH_POST_MS, bin_ms: float = DEFAULT_PSTH_BIN_MS) -> dict:
    """Peri-stimulus time histogram: the firing rate around stimuli, averaged over trials.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    stimulus_samples : array_like
        Stimulus times, as sample indices.
    sample_rate : float
        Sample rate in Hz.
    pre_ms, post_ms : float
        Window before and after each stimulus.
    bin_ms : float
        Bin width.

    Returns
    -------
    dict of numpy.ndarray
        ``time_ms``, ``rate_hz`` (mean over trials), ``se_hz``, ``trials``.
    """
    return _m.psth(_train(train), _train(stimulus_samples), sample_rate, pre_ms, post_ms, bin_ms)


def stimulus_average(data, stimulus_samples, sample_rate: float, *, pre_ms: float = 10.0, post_ms: float = 50.0, response_ms: tuple[float, float] = (2.0, 40.0), threshold_sigma: float = 3.0) -> dict:
    """Stimulus-triggered average of a signal, and the evoked response on each channel.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)`` in µV, converted to float32 on the host.
    stimulus_samples : array_like
        Stimulus times, as sample indices.
    sample_rate : float
        Sample rate in Hz.
    pre_ms, post_ms : float
        Window before and after each stimulus.
    response_ms : (float, float)
        Where the evoked response is measured, after the stimulus.
    threshold_sigma : float
        The response starts where it first exceeds this many baseline standard deviations.

    Returns
    -------
    dict
        ``time_ms``, ``mean`` / ``std`` / ``se`` ``(channels, samples)``, ``trials``; per channel
        the evoked potential: ``mep_onset_ms``, ``mep_peak_to_peak_uv``, ``mep_rms_uv``,
        ``mep_area_uv_ms``.
    """
    pre, post = round(pre_ms * sample_rate / 1e3), round(post_ms * sample_rate / 1e3)
    return _m.stimulus_average(np.ascontiguousarray(data, dtype=np.float32), _train(stimulus_samples), pre, post, sample_rate, response_ms[0], response_ms[1], threshold_sigma)


def composite_score(train, amplitudes, snr: float, total_samples: int, sample_rate: float) -> dict:
    """EMUsort's composite quality score of a unit.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    amplitudes : array_like
        The unit's spike amplitudes.
    snr : float
        The unit's signal-to-noise ratio.
    total_samples : int
        Samples in the recording.
    sample_rate : float
        Sample rate in Hz.

    Returns
    -------
    dict
        ``score`` in ``[0, 1]`` and the parts it combines.
    """
    return _m.composite_score(_train(train), np.ascontiguousarray(amplitudes, dtype=np.float32), snr, int(total_samples), sample_rate)


def firing_range(train, total_samples: int, sample_rate: float, *, bin_s: float = 0.5, percentiles: tuple[float, float] = (5.0, 95.0)) -> float:
    """Spread of a unit's firing rate: between two percentiles of its rate in short bins.

    Parameters
    ----------
    train : array_like
        Spike times, as sample indices.
    total_samples : int
        Samples in the recording.
    sample_rate : float
        Sample rate in Hz.
    bin_s : float
        Bin length.
    percentiles : (float, float)
        Lower and upper percentile.

    Returns
    -------
    float
        In Hz.
    """
    return _m.firing_range(_train(train), int(total_samples), sample_rate, bin_s, percentiles[0], percentiles[1])


def compare_trains(a, b, *, tolerance_samples: int) -> dict:
    """Matches two spike trains one to one, within a tolerance.

    Parameters
    ----------
    a, b : array_like
        Spike times, as sample indices (``a`` taken as the truth).
    tolerance_samples : int
        Spikes this close are a match.

    Returns
    -------
    dict
        ``matches``, ``false_negatives``, ``false_positives``, ``agreement``, ``precision``,
        ``recall``.

    Examples
    --------
    >>> import dspk
    >>> out = dspk.synapse.metrics.compare_trains([100, 500, 900], [102, 500, 1300], tolerance_samples=5)
    >>> out["matches"], out["false_negatives"], out["false_positives"]
    (2, 1, 1)
    """
    return _m.compare_trains(_train(a), _train(b), int(tolerance_samples))


def compare_sortings(a, b, *, delta_ms: float = DEFAULT_MATCH_DELTA_MS, threshold: float = DEFAULT_AGREEMENT_THRESHOLD) -> dict:
    """Compares two sortings unit by unit, such as a sorter's against ground truth.

    Parameters
    ----------
    a, b : Sorting
        The sortings (``a`` taken as the truth).
    delta_ms : float
        Spikes this close are a match.
    threshold : float
        Least agreement of a matched pair.

    Returns
    -------
    dict
        ``agreement`` ``(units_a, units_b)``, ``unit_ids_a``, ``unit_ids_b``, ``matched``
        (``(a, b, agreement)`` pairs) and ``well_detected`` (units of ``a``).
    """
    return _m.compare_sortings(a._native, b._native, delta_ms, threshold)


__all__ = [
    "AMPLITUDE_CUTOFF_BINS", "AMPLITUDE_CUTOFF_MIN_RATIO", "AMPLITUDE_CUTOFF_SMOOTHING", "DEFAULT_ACG_PROBABILITY", "DEFAULT_ACG_RATIO",
    "DEFAULT_AGREEMENT_THRESHOLD", "DEFAULT_CCG_PROBABILITY", "DEFAULT_CCG_RATIO", "DEFAULT_CORRELOGRAM_BIN_MS", "DEFAULT_CORRELOGRAM_WINDOW_MS",
    "DEFAULT_ISI_THRESHOLD_MS", "DEFAULT_MATCH_DELTA_MS", "DEFAULT_PSTH_BIN_MS", "DEFAULT_PSTH_POST_MS", "DEFAULT_PSTH_PRE_MS", "DEFAULT_MIN_ISI_MS", "DEFAULT_PRESENCE_BIN_S", "DEFAULT_PRESENCE_MEAN_RATE_RATIO",
    "DEFAULT_REFRACTORY_BIN_MS", "DEFAULT_REFRACTORY_WINDOW_MS",
    "acg_refractory", "amplitude_cutoff", "autocorrelogram", "bursts", "ccg_refractory", "compare_sortings", "compare_trains", "composite_score",
    "crosscorrelogram", "d_prime", "firing_range", "firing_rate", "isi_histogram", "isi_violations", "isolation_distance", "llobet_contamination",
    "presence_ratio", "psth", "refractory_violations", "silhouette", "snr", "stimulus_average",
]
