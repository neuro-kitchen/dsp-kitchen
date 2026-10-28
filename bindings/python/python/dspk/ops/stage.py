"""Pipeline stages (see :class:`dspk.ops.Pipeline`). Defaults follow scipy where an equivalent exists."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter import (
    DEFAULT_ORDER,
    FIR_DEFAULT_EDGE,
    GAUSSIAN_DEFAULT_EDGE,
    MAX_MEDIAN_WIDTH,
    MEDIAN_DEFAULT_EDGE,
    TEAGER_KAISER_DEFAULT_EDGE,
)

Stage = _ops.Stage


def scale(alpha: float, beta: float = 0.0) -> Stage:
    """``y = alpha * x + beta``."""
    return _ops.stage_scale(alpha, beta)


def subtract_baseline(baseline: float) -> Stage:
    """``y = x - baseline`` (in the signal's unit)."""
    return _ops.stage_subtract_baseline(baseline)


def clamp(min: float, max: float) -> Stage:  # noqa: A002
    """``y = clip(x, min, max)``."""
    return _ops.stage_clamp(min, max)


def iir(
    band: str,
    cutoff: float | Sequence[float],
    *,
    order: int = DEFAULT_ORDER,
    design: str = "butterworth",
    ripple_db: float | None = None,
    mode: str = "forward_backward",
    start: str = "rest",
) -> Stage:
    """IIR filter, as :func:`dspk.ops.filter.iir`."""
    cutoffs = [float(c) for c in np.atleast_1d(cutoff)]
    return _ops.stage_iir(band, cutoffs, order, design, ripple_db, mode, start)


def notch(freq: float, q: float, *, mode: str = "forward_backward", start: str = "rest") -> Stage:
    """Second-order notch, as :func:`dspk.ops.filter.notch`."""
    return _ops.stage_notch(freq, q, mode, start)


def sos(sections: np.ndarray, *, mode: str = "forward_backward", start: str = "rest") -> Stage:
    """Explicit second-order sections ``(n, 6)``, scipy layout."""
    return _ops.stage_sos(np.ascontiguousarray(sections, dtype=np.float64), mode, start)


def fir(taps, *, centered: bool = False, edge: str = FIR_DEFAULT_EDGE) -> Stage:
    """FIR filter with explicit ``taps`` (e.g. from :func:`dspk.ops.filter.firwin`): causal, or
    ``centered`` (odd number of taps). In ``Pipeline(stream=True)`` chunk edges are not carried
    across; run it on independent chunks with ``settling`` of context, which is exact for a FIR."""
    return _ops.stage_fir([float(t) for t in np.ravel(taps)], centered, edge)


def whiten(whitening) -> Stage:
    """A fitted :class:`dspk.ops.spatial.Whitening` as a stage (channels must match)."""
    return _ops.stage_whiten(whitening._native)


def laplacian(laplacian) -> Stage:  # noqa: F811 (name of the operation)
    """A :class:`dspk.ops.spatial.Laplacian` as a stage (channels must match)."""
    return _ops.stage_laplacian(laplacian._native)


def car() -> Stage:
    """Common average reference: the mean over channels subtracted at every sample."""
    return _ops.stage_car()


def cmr() -> Stage:
    """Common median reference: the median over channels subtracted at every sample."""
    return _ops.stage_cmr()


def gaussian_smooth(sigma_samples: float, *, edge: str = GAUSSIAN_DEFAULT_EDGE) -> Stage:
    """Zero-phase Gaussian smoothing, standard deviation ``sigma_samples``."""
    return _ops.stage_gaussian_smooth(sigma_samples, edge)


def median(width: int, *, edge: str = MEDIAN_DEFAULT_EDGE) -> Stage:
    """Running median over an odd ``width`` (``scipy.signal.medfilt`` with the default edge)."""
    return _ops.stage_median(width, edge)


def teager_kaiser(*, edge: str = TEAGER_KAISER_DEFAULT_EDGE) -> Stage:
    """Teager–Kaiser energy ``x[t]**2 - x[t-1] * x[t+1]``."""
    return _ops.stage_teager_kaiser(edge)


__all__ = [
    "GAUSSIAN_DEFAULT_EDGE", "MAX_MEDIAN_WIDTH", "MEDIAN_DEFAULT_EDGE", "Stage", "TEAGER_KAISER_DEFAULT_EDGE",
    "FIR_DEFAULT_EDGE", "car", "clamp", "cmr", "fir", "gaussian_smooth", "laplacian", "whiten", "iir", "median", "notch", "scale", "sos", "subtract_baseline",
    "teager_kaiser",
]
