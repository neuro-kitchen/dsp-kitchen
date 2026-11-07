"""Templates and matching: lag-maximised cosine similarity of templates, merge suggestions, and
matching pursuit of a recording against templates (on the runtime's device).

Templates are ``(units, rows, samples)``; row ``r`` of unit ``u`` is recording channel
``channels[u, r]`` (default: rows are channels ``0..rows``).
"""

from __future__ import annotations

import numpy as np

from dspk._native import synapse as _synapse

_m = _synapse.matching

DEFAULT_MIN_AMPLITUDE_SCALE: float = _m.DEFAULT_MIN_AMPLITUDE_SCALE
DEFAULT_MAX_AMPLITUDE_SCALE: float = _m.DEFAULT_MAX_AMPLITUDE_SCALE
#: µV²: rescale it with the data (it grows with the square of the amplitude unit).
DEFAULT_MIN_EXPLAINED_ENERGY: float = _m.DEFAULT_MIN_EXPLAINED_ENERGY
DEFAULT_MAX_PASSES: int = _m.DEFAULT_MAX_PASSES
DEFAULT_MAX_LAG_SAMPLES: int = 10
DEFAULT_MERGE_THRESHOLD: float = 0.9


def _templates(t) -> np.ndarray:
    t = np.ascontiguousarray(t, dtype=np.float32)
    if t.ndim == 2:
        t = t[:, None, :]
    if t.ndim != 3:
        raise ValueError(f"templates must be (units, rows, samples), got shape {t.shape}")
    return t


def _channels(c):
    return None if c is None else np.ascontiguousarray(c, dtype=np.int64)


def template_similarity(templates, *, max_lag_samples: int = DEFAULT_MAX_LAG_SAMPLES, channels=None) -> np.ndarray:
    """Cosine similarity of every pair of templates, each at its best lag.

    Parameters
    ----------
    templates : array_like
        ``(units, rows, samples)``, or ``(units, samples)`` for one row each.
    max_lag_samples : int
        Lags tried, up to this many samples each way.
    channels : array_like, optional
        ``(units, rows)``: the recording channel of each template row; by default rows are channels
        ``0..rows``. Only rows on shared channels are compared.

    Returns
    -------
    numpy.ndarray
        ``(units, units)``, in ``[-1, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> t = np.sin(np.linspace(0, 6, 60, dtype=np.float32))
    >>> sim = dspk.synapse.matching.template_similarity(np.stack([t, np.roll(t, 3)]))
    >>> bool(sim[0, 1] > 0.9)
    True
    """
    return _m.template_similarity(_templates(templates), max_lag_samples=max_lag_samples, channels=_channels(channels))


def suggest_merges(templates, *, max_lag_samples: int = DEFAULT_MAX_LAG_SAMPLES, threshold: float = DEFAULT_MERGE_THRESHOLD, channels=None) -> list[tuple[int, int, float]]:
    """Pairs of units whose templates are alike enough to be one unit.

    Parameters
    ----------
    templates : array_like
        ``(units, rows, samples)``, or ``(units, samples)`` for one row each.
    max_lag_samples : int
        Lags tried, up to this many samples each way.
    threshold : float
        Least similarity of a suggested pair.
    channels : array_like, optional
        ``(units, rows)``: the recording channel of each template row; by default rows are channels
        ``0..rows``. Only rows on shared channels are compared.

    Returns
    -------
    list of (int, int, float)
        ``(i, j, similarity)`` of each pair.
    """
    return _m.suggest_merges(_templates(templates), max_lag_samples=max_lag_samples, threshold=threshold, channels=_channels(channels))


def matching_pursuit(
    data,
    templates,
    *,
    min_amplitude_scale: float = DEFAULT_MIN_AMPLITUDE_SCALE,
    max_amplitude_scale: float = DEFAULT_MAX_AMPLITUDE_SCALE,
    min_explained_energy: float = DEFAULT_MIN_EXPLAINED_ENERGY,
    max_passes: int = DEFAULT_MAX_PASSES,
    channels=None,
    runtime: str | None = None,
) -> dict:
    """Finds spikes by subtracting the best-fitting template, pass after pass, on the device.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)`` in µV, converted to float32 on the host.
    templates : array_like
        ``(units, rows, samples)``, or ``(units, samples)`` for one row each.
    min_amplitude_scale, max_amplitude_scale : float
        Range a template may be scaled by to fit.
    min_explained_energy : float
        Least energy (µV²) a fit must remove to count; rescale it with the data's unit.
    max_passes : int
        Passes over the data.
    channels : array_like, optional
        ``(units, rows)``: the recording channel of each template row; by default rows are channels
        ``0..rows``. Only rows on shared channels are compared.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available`; by default the current one.

    Returns
    -------
    dict of numpy.ndarray
        ``samples`` (troughs), ``units``, ``amplitudes`` (scales), ``scores``.
    """
    return _m.matching_pursuit(np.ascontiguousarray(data, dtype=np.float32), _templates(templates), min_amplitude_scale=min_amplitude_scale,
                               max_amplitude_scale=max_amplitude_scale, min_explained_energy=min_explained_energy, max_passes=max_passes,
                               channels=_channels(channels), runtime=runtime)


__all__ = [
    "DEFAULT_MAX_AMPLITUDE_SCALE", "DEFAULT_MAX_LAG_SAMPLES", "DEFAULT_MAX_PASSES", "DEFAULT_MERGE_THRESHOLD", "DEFAULT_MIN_AMPLITUDE_SCALE",
    "DEFAULT_MIN_EXPLAINED_ENERGY", "matching_pursuit", "suggest_merges", "template_similarity",
]
