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
    """Lowers the sample rate by an integer factor, behind an anti-alias filter.

    As ``scipy.signal.decimate(x, q, ftype=ftype, zero_phase=True)``.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    q : int
        Factor: every ``q``-th sample is kept, ``ceil(samples / q)`` samples out.
    ftype : {"iir", "fir"}
        ``"iir"``: a zero-phase Chebyshev I of ``order`` and ``ripple_db``, cut at
        :data:`DECIMATE_IIR_CUTOFF` of the new Nyquist. ``"fir"``: a Hamming ``firwin`` of
        ``taps_per_factor * q + 1`` taps, applied as :func:`resample_poly` does.
    order : int
        Order of the IIR filter.
    ripple_db : float
        Passband ripple of the IIR filter, in dB.
    taps_per_factor : int
        FIR taps per unit of ``q``.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        ``(channels, ceil(samples / q))``, float32.

    Raises
    ------
    ValueError
        For ``q`` below 1, an unknown ``ftype``, or ``data`` not 2-D.
    RuntimeError
        When the runtime has no device.

    Notes
    -----
    With ``"iir"``, samples near the ends can differ slightly from scipy's, which pads the edges
    differently; the interior matches.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> lfp = dspk.ops.resample.decimate(x, 30)
    >>> lfp.shape
    (4, 1000)
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
    """Changes the sample rate by ``up / down`` with a polyphase FIR filter.

    As ``scipy.signal.resample_poly``: up-sample by ``up``, low-pass with a Kaiser-windowed FIR,
    down-sample by ``down``.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    up : int
        Up-sampling factor.
    down : int
        Down-sampling factor.
    beta : float
        Kaiser window shape (larger: less ripple, wider transition).
    edge : {"zeros", "odd", "reflect", "nearest"}
        What the samples past each end are taken to be.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        ``(channels, ceil(samples * up / down))``, float32.

    Raises
    ------
    ValueError
        For ``up`` or ``down`` below 1, an unknown ``edge``, or ``data`` not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> y = dspk.ops.resample.resample_poly(x, 2, 3)
    >>> y.shape
    (4, 20000)
    """
    return _resample.resample_poly(_channels(data), up, down, beta=beta, edge=edge, runtime=runtime)


__all__ = [
    "DECIMATE_DEFAULT_ORDER", "DECIMATE_DEFAULT_RIPPLE_DB", "DECIMATE_FIR_TAPS_PER_FACTOR", "DECIMATE_IIR_CUTOFF",
    "RESAMPLE_POLY_BETA", "RESAMPLE_POLY_DEFAULT_EDGE", "RESAMPLE_POLY_HALF_LEN_PER_RATE", "decimate", "resample_poly",
]
