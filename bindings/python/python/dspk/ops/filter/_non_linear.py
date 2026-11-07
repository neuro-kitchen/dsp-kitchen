"""Median and Teager–Kaiser on the device."""

from __future__ import annotations

import numpy as np

from dspk.ops.filter._common import _channels, _filter

MEDIAN_DEFAULT_EDGE: str = _filter.MEDIAN_DEFAULT_EDGE
TEAGER_KAISER_DEFAULT_EDGE: str = _filter.TEAGER_KAISER_DEFAULT_EDGE
MAX_MEDIAN_WIDTH: int = _filter.MAX_MEDIAN_WIDTH


def median(data, width: int, *, edge: str = MEDIAN_DEFAULT_EDGE, runtime: str | None = None) -> np.ndarray:
    """Running median along time, as ``scipy.signal.medfilt`` (with ``edge="zeros"``).

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    width : int
        Window length in samples, odd, at most :data:`MAX_MEDIAN_WIDTH`.
    edge : {"zeros", "odd", "reflect", "nearest"}
        What the samples past each end are taken to be.
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
        For an even or too large ``width``, an unknown ``edge``, or ``data`` not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> baseline = dspk.ops.filter.median(x, 31)
    """
    return _filter.median(_channels(data), width, edge=edge, runtime=runtime)


def teager_kaiser(data, *, edge: str = TEAGER_KAISER_DEFAULT_EDGE, runtime: str | None = None) -> np.ndarray:
    """Teager–Kaiser energy along time: ``x[t]**2 - x[t-1] * x[t+1]``.

    It is large where the signal is both large and fast, so spikes stand out from slow waves.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    edge : {"zeros", "odd", "reflect", "nearest"}
        What the neighbours of the first and last samples are taken to be.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        The energy ``(channels, samples)``, float32.

    Raises
    ------
    ValueError
        For an unknown ``edge`` or ``data`` not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> energy = dspk.ops.filter.teager_kaiser(x)
    """
    return _filter.teager_kaiser(_channels(data), edge=edge, runtime=runtime)


__all__ = ["MAX_MEDIAN_WIDTH", "MEDIAN_DEFAULT_EDGE", "TEAGER_KAISER_DEFAULT_EDGE", "median", "teager_kaiser"]
