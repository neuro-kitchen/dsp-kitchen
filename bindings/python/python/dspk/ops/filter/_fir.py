"""FIR filters and Gaussian smoothing on the device, and ``firwin`` design."""

from __future__ import annotations

import numpy as np

from dspk.ops.filter._common import _channels, _filter

FIR_DEFAULT_EDGE: str = _filter.FIR_DEFAULT_EDGE
GAUSSIAN_DEFAULT_EDGE: str = _filter.GAUSSIAN_DEFAULT_EDGE
GAUSSIAN_TRUNCATE: float = _filter.GAUSSIAN_TRUNCATE


def firwin(
    num_taps: int,
    cutoff: float,
    *,
    sample_rate: float | None = None,
    window: str = "hamming",
    beta: float | None = None,
) -> np.ndarray:
    """Low-pass FIR taps with unit DC gain (``scipy.signal.firwin`` with one cutoff).

    ``cutoff`` is in Hz when ``sample_rate`` is given, else a fraction of Nyquist in ``(0, 1]``.
    ``window``: ``"hamming"`` or ``"kaiser"`` (needs ``beta``).
    """
    return _filter.firwin(num_taps, cutoff, sample_rate=sample_rate, window=window, beta=beta)


def fir(
    data,
    taps,
    *,
    centered: bool = False,
    edge: str = FIR_DEFAULT_EDGE,
    runtime: str | None = None,
) -> np.ndarray:
    """FIR filtering of ``data`` ``(channels, samples)`` with ``taps`` on the device.

    ``centered=False``: causal ``y[t] = sum(taps[k] * x[t - k])``, as
    ``scipy.signal.lfilter(taps, 1, x)`` with ``edge="zeros"``. ``centered=True``: an odd number of
    taps centred on each sample (zero phase for symmetric taps). Samples past the ends come from
    ``edge`` (``"zeros"``, ``"odd"``, ``"reflect"``, ``"nearest"``).
    """
    taps = np.ascontiguousarray(taps, dtype=np.float32).ravel()
    return _filter.fir(_channels(data), taps, centered=centered, edge=edge, runtime=runtime)


def gaussian_smooth(
    data,
    sigma_samples: float,
    *,
    edge: str = GAUSSIAN_DEFAULT_EDGE,
    runtime: str | None = None,
) -> np.ndarray:
    """Zero-phase Gaussian smoothing, standard deviation ``sigma_samples``, cut at
    :data:`GAUSSIAN_TRUNCATE` sigma (``scipy.ndimage.gaussian_filter1d`` with ``mode=edge``)."""
    return _filter.gaussian_smooth(_channels(data), sigma_samples, edge=edge, runtime=runtime)


__all__ = ["FIR_DEFAULT_EDGE", "GAUSSIAN_DEFAULT_EDGE", "GAUSSIAN_TRUNCATE", "fir", "firwin", "gaussian_smooth"]
