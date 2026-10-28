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

_filter = _ops.filter

DEFAULT_ORDER: int = _filter.DEFAULT_ORDER
SETTLING_TOLERANCE: float = _filter.SETTLING_TOLERANCE


def _channels(data) -> np.ndarray:
    array = np.ascontiguousarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"data must be 2-D (channels, samples), got shape {array.shape}")
    return array


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
    """Second-order sections ``(sections, 6)`` of an IIR design, scipy layout ``[b, a]``.

    ``band``: ``"lowpass"``, ``"highpass"`` (one ``cutoff`` in Hz), ``"bandpass"``, ``"bandstop"``
    (``(low, high)`` in Hz). ``design``: ``"butterworth"``, ``"chebyshev1"`` (needs ``ripple_db``),
    ``"bessel"``.
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
    """Designs an IIR filter (as :func:`sos`) and applies it to ``data`` on the device."""
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
    """Second-order notch at ``freq`` Hz, quality factor ``q`` (``scipy.signal.iirnotch``)."""
    return _filter.notch(_channels(data), sample_rate, freq, q, mode=mode, start=start, runtime=runtime)


def sosfilt(
    data,
    sos: np.ndarray,
    *,
    mode: str = "forward_backward",
    start: str = "rest",
    runtime: str | None = None,
) -> np.ndarray:
    """Applies second-order sections ``sos`` ``(sections, 6)`` to ``data`` on the device."""
    sections = np.ascontiguousarray(sos, dtype=np.float64)
    return _filter.sosfilt(_channels(data), sections, mode=mode, start=start, runtime=runtime)


def settling_samples(sos: np.ndarray) -> int:
    """Samples until the impulse response of ``sos`` decays below :data:`SETTLING_TOLERANCE`."""
    return _filter.settling_samples(np.ascontiguousarray(sos, dtype=np.float64))


__all__ = ["DEFAULT_ORDER", "SETTLING_TOLERANCE", "iir", "notch", "settling_samples", "sos", "sosfilt"]
