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
    """Scales and shifts: ``y = alpha * x + beta``.

    Parameters
    ----------
    alpha : float
        Factor.
    beta : float
        Offset, in the signal's unit.

    Returns
    -------
    Stage
    """
    return _ops.stage_scale(alpha, beta)


def subtract_baseline(baseline: float) -> Stage:
    """Subtracts a constant: ``y = x - baseline``.

    Parameters
    ----------
    baseline : float
        In the signal's unit.

    Returns
    -------
    Stage
    """
    return _ops.stage_subtract_baseline(baseline)


def clamp(min: float, max: float) -> Stage:  # noqa: A002
    """Clips to a range: ``y = clip(x, min, max)``.

    Parameters
    ----------
    min : float
        Lower bound.
    max : float
        Upper bound, at least ``min``.

    Returns
    -------
    Stage

    Raises
    ------
    ValueError
        When ``min`` is above ``max``.
    """
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
    """An IIR filter, as :func:`dspk.ops.filter.iir`; the pipeline gives the sample rate.

    Parameters
    ----------
    band : {"lowpass", "highpass", "bandpass", "bandstop"}
        The band the filter passes (or stops).
    cutoff : float or (float, float)
        Cutoff in Hz: one, or ``(low, high)`` for the band filters.
    order : int
        Filter order.
    design : {"butterworth", "chebyshev1", "bessel"}
        The filter family.
    ripple_db : float, optional
        Passband ripple in dB; ``"chebyshev1"`` needs it.
    mode : {"forward_backward", "forward"}
        ``"forward_backward"``: zero phase (windowed runs only). ``"forward"``: causal, the
        mode ``stream=True`` needs.
    start : {"rest", "steady_state"}
        Where a ``"forward"`` pass starts, as in :func:`dspk.ops.filter.iir`.

    Returns
    -------
    Stage

    Raises
    ------
    ValueError
        For an invalid design, ``mode`` or ``start``.
    """
    cutoffs = [float(c) for c in np.atleast_1d(cutoff)]
    return _ops.stage_iir(band, cutoffs, order, design, ripple_db, mode, start)


def notch(freq: float, q: float, *, mode: str = "forward_backward", start: str = "rest") -> Stage:
    """A second-order notch, as :func:`dspk.ops.filter.notch`.

    Parameters
    ----------
    freq : float
        Frequency to remove, in Hz.
    q : float
        Quality factor: the notch is ``freq / q`` Hz wide at -3 dB.
    mode : {"forward_backward", "forward"}
        ``"forward_backward"``: zero phase (windowed runs only). ``"forward"``: causal, the
        mode ``stream=True`` needs.
    start : {"rest", "steady_state"}
        Where a ``"forward"`` pass starts, as in :func:`dspk.ops.filter.iir`.

    Returns
    -------
    Stage
    """
    return _ops.stage_notch(freq, q, mode, start)


def sos(sections: np.ndarray, *, mode: str = "forward_backward", start: str = "rest") -> Stage:
    """Second-order sections you already have, such as from ``scipy.signal``.

    Parameters
    ----------
    sections : numpy.ndarray
        ``(sections, 6)`` in scipy's layout.
    mode : {"forward_backward", "forward"}
        ``"forward_backward"``: zero phase (windowed runs only). ``"forward"``: causal, the
        mode ``stream=True`` needs.
    start : {"rest", "steady_state"}
        Where a ``"forward"`` pass starts, as in :func:`dspk.ops.filter.iir`.

    Returns
    -------
    Stage
    """
    return _ops.stage_sos(np.ascontiguousarray(sections, dtype=np.float64), mode, start)


def fir(taps, *, centered: bool = False, edge: str = FIR_DEFAULT_EDGE) -> Stage:
    """An FIR filter with explicit taps, causal or centred.

    Parameters
    ----------
    taps : array_like
        Filter taps (1-D), e.g. from :func:`dspk.ops.filter.firwin`.
    centered : bool
        ``False``: causal. ``True``: the taps (an odd number) centred on each sample.
    edge : {"zeros", "odd", "reflect", "nearest"}
        What the samples past each end are taken to be.

    Returns
    -------
    Stage

    Raises
    ------
    ValueError
        For empty ``taps`` or an even number of centred taps.

    Notes
    -----
    In ``Pipeline(stream=True)`` the taps' history is not carried from one chunk to the next: run
    it on independent chunks with :attr:`Pipeline.settling` of context, which is exact for a FIR.
    """
    return _ops.stage_fir([float(t) for t in np.ravel(taps)], centered, edge)


def whiten(whitening) -> Stage:
    """A fitted :class:`dspk.ops.spatial.Whitening` as a stage.

    Parameters
    ----------
    whitening : dspk.ops.spatial.Whitening
        Fitted on as many channels as the pipeline's input.

    Returns
    -------
    Stage
    """
    return _ops.stage_whiten(whitening._native)


def laplacian(laplacian) -> Stage:  # noqa: F811 (name of the operation)
    """A :class:`dspk.ops.spatial.Laplacian` as a stage.

    Parameters
    ----------
    laplacian : dspk.ops.spatial.Laplacian
        Made for as many channels as the pipeline's input.

    Returns
    -------
    Stage
    """
    return _ops.stage_laplacian(laplacian._native)


def reorder_channels(order: Sequence[int]) -> Stage:
    """The channels in a new order: output channel ``i`` is input channel ``order[i]``.

    Every channel is used once, so the channel count stays. To drop channels, use
    :func:`dspk.ops.layout.remap`, or a ``remap`` node right after ``read`` or before ``write`` in a
    pipeline spec.

    Parameters
    ----------
    order : sequence of int
        A permutation of ``0..channels``.

    Returns
    -------
    Stage

    Raises
    ------
    ValueError
        When ``order`` is not a permutation.
    """
    return _ops.stage_reorder_channels([int(c) for c in order])


def remap(map) -> Stage:  # noqa: A002
    """A channel map that keeps every channel, as a stage (:func:`reorder_channels` of its order).

    Parameters
    ----------
    map : dict, list of dict, str or os.PathLike
        A channel map: a ``{"base", "pairs"}`` dict (or a file's ``{"remap": {...}}``), a list of
        ``{"from", "to", "name"}`` pairs, or the path of a ``.toml`` / ``.json`` map file.

    Returns
    -------
    Stage

    Raises
    ------
    ValueError
        When the map is invalid or drops channels.
    """
    from dspk.ops.layout import map_order

    order, _ = map_order(map)
    return reorder_channels(order)


def car() -> Stage:
    """Common average reference, as :func:`dspk.ops.spatial.car`.

    Returns
    -------
    Stage
    """
    return _ops.stage_car()


def cmr() -> Stage:
    """Common median reference, as :func:`dspk.ops.spatial.cmr`.

    Returns
    -------
    Stage
    """
    return _ops.stage_cmr()


def gaussian_smooth(sigma_samples: float, *, edge: str = GAUSSIAN_DEFAULT_EDGE) -> Stage:
    """Zero-phase Gaussian smoothing, as :func:`dspk.ops.filter.gaussian_smooth`.

    Parameters
    ----------
    sigma_samples : float
        Standard deviation, in samples.
    edge : {"zeros", "odd", "reflect", "nearest"}
        What the samples past each end are taken to be.

    Returns
    -------
    Stage

    Raises
    ------
    ValueError
        For a non-positive ``sigma_samples``.
    """
    return _ops.stage_gaussian_smooth(sigma_samples, edge)


def median(width: int, *, edge: str = MEDIAN_DEFAULT_EDGE) -> Stage:
    """Running median, as :func:`dspk.ops.filter.median`.

    Parameters
    ----------
    width : int
        Window length in samples, odd, at most :data:`MAX_MEDIAN_WIDTH`.
    edge : {"zeros", "odd", "reflect", "nearest"}
        What the samples past each end are taken to be.

    Returns
    -------
    Stage

    Raises
    ------
    ValueError
        For an even or too large ``width``.
    """
    return _ops.stage_median(width, edge)


def teager_kaiser(*, edge: str = TEAGER_KAISER_DEFAULT_EDGE) -> Stage:
    """Teager–Kaiser energy, as :func:`dspk.ops.filter.teager_kaiser`.

    Parameters
    ----------
    edge : {"zeros", "odd", "reflect", "nearest"}
        What the samples past each end are taken to be.

    Returns
    -------
    Stage
    """
    return _ops.stage_teager_kaiser(edge)


LOG10_FLOOR: float = _ops.LOG10_FLOOR
"""Smallest magnitude ``log10`` takes the logarithm of (zeros give ``log10(LOG10_FLOOR)``)."""


def rectify() -> Stage:
    """Full-wave rectification, ``|x|``; a low-pass after it gives an envelope.

    Returns
    -------
    Stage
    """
    return _ops.stage_elementwise("abs")


def square() -> Stage:
    """Instantaneous power, ``x**2``; a low-pass and :func:`sqrt` after it give a moving RMS.

    Returns
    -------
    Stage
    """
    return _ops.stage_elementwise("square")


def sqrt() -> Stage:
    """Square root, ``sqrt(x)``, with negative values taken as 0.

    Returns
    -------
    Stage
    """
    return _ops.stage_elementwise("sqrt")


def log10() -> Stage:
    """``log10(|x|)``, with magnitudes below :data:`LOG10_FLOOR` taken as it.

    Returns
    -------
    Stage
    """
    return _ops.stage_elementwise("log10")


__all__ = [
    "GAUSSIAN_DEFAULT_EDGE", "MAX_MEDIAN_WIDTH", "MEDIAN_DEFAULT_EDGE", "Stage", "TEAGER_KAISER_DEFAULT_EDGE",
    "FIR_DEFAULT_EDGE", "car", "clamp", "cmr", "fir", "gaussian_smooth", "laplacian", "whiten", "iir", "median", "notch", "scale", "sos", "subtract_baseline",
    "remap", "reorder_channels", "teager_kaiser", "LOG10_FLOOR", "rectify", "square", "sqrt", "log10",
]
