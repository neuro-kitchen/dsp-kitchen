"""Sample-rate changes on the device, with ``scipy.signal.decimate`` / ``resample_poly`` semantics."""

from __future__ import annotations

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_resample = _ops.resample

DECIMATE_DEFAULT_ORDER: int = _resample.DECIMATE_DEFAULT_ORDER
DECIMATE_DEFAULT_RIPPLE_DB: float = _resample.DECIMATE_DEFAULT_RIPPLE_DB
DECIMATE_IIR_CUTOFF: float = _resample.DECIMATE_IIR_CUTOFF
DECIMATE_FIR_TAPS_PER_FACTOR: int = _resample.DECIMATE_FIR_TAPS_PER_FACTOR
RESAMPLE_POLY_BETA: float = _resample.RESAMPLE_POLY_BETA
RESAMPLE_POLY_HALF_LEN_PER_RATE: int = _resample.RESAMPLE_POLY_HALF_LEN_PER_RATE
RESAMPLE_POLY_DEFAULT_EDGE: str = _resample.RESAMPLE_POLY_DEFAULT_EDGE


def decimate(
    data,
    q: int,
    *,
    ftype: str = "iir",
    order: int = DECIMATE_DEFAULT_ORDER,
    ripple_db: float = DECIMATE_DEFAULT_RIPPLE_DB,
    taps_per_factor: int = DECIMATE_FIR_TAPS_PER_FACTOR,
    runtime: str | None = None,
) -> np.ndarray:
    """Down-samples ``data`` ``(channels, samples)`` by ``q`` behind an anti-alias filter, as
    ``scipy.signal.decimate(x, q, ftype=ftype, zero_phase=True)``; ``ceil(samples / q)`` samples out.

    ``ftype="iir"``: Chebyshev I of ``order`` / ``ripple_db`` at :data:`DECIMATE_IIR_CUTOFF` of the
    new Nyquist, zero phase (samples near the ends can differ slightly from scipy's padding).
    ``ftype="fir"``: Hamming ``firwin`` of ``taps_per_factor * q + 1`` taps through ``resample_poly``.
    """
    return _resample.decimate(
        _channels(data), q, ftype=ftype, order=order, ripple_db=ripple_db, taps_per_factor=taps_per_factor, runtime=runtime
    )


def resample_poly(
    data,
    up: int,
    down: int,
    *,
    beta: float = RESAMPLE_POLY_BETA,
    edge: str = RESAMPLE_POLY_DEFAULT_EDGE,
    runtime: str | None = None,
) -> np.ndarray:
    """Resamples ``data`` ``(channels, samples)`` by ``up / down`` (polyphase FIR, Kaiser ``beta``
    window), as ``scipy.signal.resample_poly``; samples past the ends come from ``edge``."""
    return _resample.resample_poly(_channels(data), up, down, beta=beta, edge=edge, runtime=runtime)


__all__ = [
    "DECIMATE_DEFAULT_ORDER", "DECIMATE_DEFAULT_RIPPLE_DB", "DECIMATE_FIR_TAPS_PER_FACTOR", "DECIMATE_IIR_CUTOFF",
    "RESAMPLE_POLY_BETA", "RESAMPLE_POLY_DEFAULT_EDGE", "RESAMPLE_POLY_HALF_LEN_PER_RATE", "decimate", "resample_poly",
]
