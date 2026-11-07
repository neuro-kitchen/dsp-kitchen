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
    """Low-pass FIR taps with unit gain at DC, as ``scipy.signal.firwin`` with one cutoff.

    Parameters
    ----------
    num_taps : int
        Number of taps (odd for a centred, zero-phase filter).
    cutoff : float
        Cutoff in Hz when ``sample_rate`` is given, else a fraction of Nyquist in ``(0, 1]``.
    sample_rate : float, optional
        Sample rate in Hz.
    window : {"hamming", "kaiser"}
        The window shaping the taps.
    beta : float, optional
        Kaiser shape parameter; ``"kaiser"`` needs it, ``"hamming"`` takes none.

    Returns
    -------
    numpy.ndarray
        Taps ``(num_taps,)``.

    Raises
    ------
    ValueError
        For ``num_taps`` below 1, a cutoff outside ``(0, Nyquist]``, a non-positive
        ``sample_rate``, an unknown ``window``, or ``beta`` given to (or missing from) a window.

    Examples
    --------
    >>> import dspk
    >>> taps = dspk.ops.filter.firwin(101, 250.0, sample_rate=30_000.0)
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
    """Filters with FIR taps on the device.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    taps : array_like
        Filter taps (1-D).
    centered : bool
        ``False``: causal, ``y[t] = sum(taps[k] * x[t - k])``, as
        ``scipy.signal.lfilter(taps, 1, x)`` with ``edge="zeros"``. ``True``: the taps (an odd
        number) centred on each sample, zero phase for symmetric taps.
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
        For empty ``taps``, an even number of centred taps, an unknown ``edge``, or ``data``
        not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> taps = dspk.ops.filter.firwin(101, 250.0, sample_rate=30_000.0)
    >>> lfp = dspk.ops.filter.fir(x, taps, centered=True)
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
    """Zero-phase Gaussian smoothing, as ``scipy.ndimage.gaussian_filter1d`` along time.

    The kernel is cut at :data:`GAUSSIAN_TRUNCATE` standard deviations.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    sigma_samples : float
        Standard deviation of the Gaussian, in samples.
    edge : {"zeros", "odd", "reflect", "nearest"}
        What the samples past each end are taken to be (scipy's ``mode``).
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
        For a non-positive ``sigma_samples``, an unknown ``edge``, or ``data`` not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> smooth = dspk.ops.filter.gaussian_smooth(np.abs(x), 30.0)
    """
    return _filter.gaussian_smooth(_channels(data), sigma_samples, edge=edge, runtime=runtime)


__all__ = ["FIR_DEFAULT_EDGE", "GAUSSIAN_DEFAULT_EDGE", "GAUSSIAN_TRUNCATE", "fir", "firwin", "gaussian_smooth"]
