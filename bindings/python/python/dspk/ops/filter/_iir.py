"""IIR filters as second-order sections, with ``scipy.signal`` semantics, run on the device.

``mode="forward_backward"`` (default) is zero phase like ``scipy.signal.sosfiltfilt``: both edges
odd-padded by :func:`settling_samples` (at most ``samples - 1``), each pass started from the steady
state of its first sample. ``mode="forward"`` is causal like ``sosfilt``: started at rest, or from
``sosfilt_zi(sos) * x[0]`` with ``start="steady_state"``.

Data is ``(channels, samples)``; values are filtered as ``float32`` on the device.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_filter = _ops.filter

DEFAULT_ORDER: int = _filter.DEFAULT_ORDER
SETTLING_TOLERANCE: float = _filter.SETTLING_TOLERANCE


def _cutoff(cutoff: float | Sequence[float]) -> list[float]:
    return [float(c) for c in np.atleast_1d(cutoff)]


def sos(
    band: str,
    cutoff: float | Sequence[float],
    sample_rate: float,
    *,
    order: int = DEFAULT_ORDER,
    design: str = "butterworth",
    ripple_db: float | None = None,
) -> np.ndarray:
    """Second-order sections of an IIR filter design, in scipy's layout.

    Parameters
    ----------
    band : {"lowpass", "highpass", "bandpass", "bandstop"}
        The band the filter passes (or stops).
    cutoff : float or (float, float)
        Cutoff in Hz: one for ``"lowpass"`` and ``"highpass"``, ``(low, high)`` for
        ``"bandpass"`` and ``"bandstop"``.
    sample_rate : float
        Sample rate in Hz.
    order : int
        Filter order.
    design : {"butterworth", "chebyshev1", "bessel"}
        The filter family.
    ripple_db : float, optional
        Passband ripple in dB; ``"chebyshev1"`` needs it, the other designs take none.

    Returns
    -------
    numpy.ndarray
        Sections ``(sections, 6)``, each row ``[b0, b1, b2, a0, a1, a2]`` as
        ``scipy.signal.butter(..., output="sos")`` gives.

    Raises
    ------
    ValueError
        For an unknown ``band`` or ``design``, the wrong number of cutoffs, or ``ripple_db``
        given to (or missing from) a design.

    Examples
    --------
    >>> import dspk
    >>> sections = dspk.ops.filter.sos("bandpass", (300.0, 6000.0), 30_000.0, order=4)
    """
    return _filter.sos(band, _cutoff(cutoff), sample_rate, order=order, design=design, ripple_db=ripple_db)


def iir(
    data,
    sample_rate: float,
    band: str,
    cutoff: float | Sequence[float],
    *,
    order: int = DEFAULT_ORDER,
    design: str = "butterworth",
    ripple_db: float | None = None,
    mode: str = "forward_backward",
    start: str = "rest",
    runtime: str | None = None,
) -> np.ndarray:
    """Designs an IIR filter and applies it on the device.

    The design is :func:`sos`'s; filtering is :func:`sosfilt`'s.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    sample_rate : float
        Sample rate of ``data`` in Hz.
    band : {"lowpass", "highpass", "bandpass", "bandstop"}
        The band the filter passes (or stops).
    cutoff : float or (float, float)
        Cutoff in Hz: one for ``"lowpass"`` and ``"highpass"``, ``(low, high)`` for
        ``"bandpass"`` and ``"bandstop"``.
    order : int
        Filter order (a zero-phase run applies it twice).
    design : {"butterworth", "chebyshev1", "bessel"}
        The filter family.
    ripple_db : float, optional
        Passband ripple in dB; ``"chebyshev1"`` needs it.
    mode : {"forward_backward", "forward"}
        ``"forward_backward"``: zero phase, as ``scipy.signal.sosfiltfilt`` (the magnitude
        response is applied twice). ``"forward"``: causal, one pass, as ``scipy.signal.sosfilt``.
    start : {"rest", "steady_state"}
        Where a ``"forward"`` pass starts: at rest, as if the signal were zero before the first
        sample (``sosfilt`` without ``zi``), or from the steady state of the first sample
        (``zi = sosfilt_zi(sos) * x[0]``: no step at the start). Forward-backward passes always
        start from the steady state.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        The filtered signal ``(channels, samples)``, float32.

    Raises
    ------
    ValueError
        For an invalid design (see :func:`sos`), ``mode`` or ``start``, or ``data`` not 2-D.
    RuntimeError
        When the runtime has no device.

    Notes
    -----
    Zero-phase runs pad each end by :func:`settling_samples` (odd reflection), where
    ``scipy.signal.sosfiltfilt`` pads ``3 * (2 * len(sos) + 1)`` samples by default: the
    interior is the same, the first and last samples can differ. ``sosfiltfilt(sos, x,
    padlen=settling_samples(sos))`` matches to float32 precision.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> spikes = dspk.ops.filter.iir(x, 30_000.0, "highpass", 300.0)
    >>> spikes.shape
    (4, 30000)
    """
    return _filter.iir(
        _channels(data), sample_rate, band, _cutoff(cutoff),
        order=order, design=design, ripple_db=ripple_db, mode=mode, start=start, runtime=runtime,
    )


def notch(
    data,
    sample_rate: float,
    freq: float,
    q: float,
    *,
    mode: str = "forward_backward",
    start: str = "rest",
    runtime: str | None = None,
) -> np.ndarray:
    """Removes one frequency (a second-order notch), as ``scipy.signal.iirnotch``.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    sample_rate : float
        Sample rate of ``data`` in Hz.
    freq : float
        Frequency to remove, in Hz (mains hum: 50 or 60).
    q : float
        Quality factor: the notch is ``freq / q`` Hz wide at -3 dB; larger is narrower.
    mode : {"forward_backward", "forward"}
        ``"forward_backward"``: zero phase, as ``scipy.signal.sosfiltfilt`` (the magnitude
        response is applied twice). ``"forward"``: causal, one pass, as ``scipy.signal.sosfilt``.
    start : {"rest", "steady_state"}
        Where a ``"forward"`` pass starts: at rest, as if the signal were zero before the first
        sample (``sosfilt`` without ``zi``), or from the steady state of the first sample
        (``zi = sosfilt_zi(sos) * x[0]``: no step at the start). Forward-backward passes always
        start from the steady state.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        The filtered signal ``(channels, samples)``, float32.

    Raises
    ------
    ValueError
        When ``data`` is not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> clean = dspk.ops.filter.notch(x, 30_000.0, 60.0, 30.0)
    """
    return _filter.notch(_channels(data), sample_rate, freq, q, mode=mode, start=start, runtime=runtime)


def sosfilt(
    data,
    sos: np.ndarray,
    *,
    mode: str = "forward_backward",
    start: str = "rest",
    runtime: str | None = None,
) -> np.ndarray:
    """Applies second-order sections on the device.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    sos : numpy.ndarray
        Sections ``(sections, 6)`` in scipy's layout, from :func:`sos` or ``scipy.signal``.
    mode : {"forward_backward", "forward"}
        ``"forward_backward"``: zero phase, as ``scipy.signal.sosfiltfilt`` (the magnitude
        response is applied twice). ``"forward"``: causal, one pass, as ``scipy.signal.sosfilt``.
    start : {"rest", "steady_state"}
        Where a ``"forward"`` pass starts: at rest, as if the signal were zero before the first
        sample (``sosfilt`` without ``zi``), or from the steady state of the first sample
        (``zi = sosfilt_zi(sos) * x[0]``: no step at the start). Forward-backward passes always
        start from the steady state.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        The filtered signal ``(channels, samples)``, float32.

    Raises
    ------
    ValueError
        For an invalid ``mode`` or ``start``, or ``data`` not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> sections = dspk.ops.filter.sos("lowpass", 250.0, 30_000.0)
    >>> lfp = dspk.ops.filter.sosfilt(x, sections)
    """
    sections = np.ascontiguousarray(sos, dtype=np.float64)
    return _filter.sosfilt(_channels(data), sections, mode=mode, start=start, runtime=runtime)


def settling_samples(sos: np.ndarray) -> int:
    """Samples after which the impulse response of ``sos`` has settled.

    The first sample ``t`` from which the rest of the response holds at most
    :data:`SETTLING_TOLERANCE` of its total magnitude (``sum(|h[t:]|) <= tolerance *
    sum(|h|)``). It is how far a filter "remembers": the padding zero-phase runs use and the
    halo windowed runs need.

    Parameters
    ----------
    sos : numpy.ndarray
        Sections ``(sections, 6)`` in scipy's layout.

    Returns
    -------
    int
        A number of samples.

    Examples
    --------
    >>> import dspk
    >>> n = dspk.ops.filter.settling_samples(dspk.ops.filter.sos("highpass", 300.0, 30_000.0))
    """
    return _filter.settling_samples(np.ascontiguousarray(sos, dtype=np.float64))


__all__ = ["DEFAULT_ORDER", "SETTLING_TOLERANCE", "iir", "notch", "settling_samples", "sos", "sosfilt"]
