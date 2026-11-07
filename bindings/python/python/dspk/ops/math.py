"""Per-channel statistics on the device, and window designs."""

from __future__ import annotations

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_math = _ops.math

MAD_TO_SIGMA: float = _math.MAD_TO_SIGMA


def channel_mad(data, *, runtime: str | None = None) -> np.ndarray:
    """Median absolute deviation of every channel, ``median(|x - median(x)|)``.

    Unscaled: divide by :data:`MAD_TO_SIGMA` for a standard deviation (SpikeInterface's
    ``get_noise_levels(method="mad")`` does). Medians of an even count are NumPy's.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        One value per channel ``(channels,)``.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> mad = dspk.ops.math.channel_mad(x)
    """
    return _math.channel_mad(_channels(data), runtime=runtime)


def channel_mean_std(data, *, runtime: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation of every channel.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    mean : numpy.ndarray
        ``(channels,)``.
    std : numpy.ndarray
        ``(channels,)``, population (``ddof=0``).

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> mean, std = dspk.ops.math.channel_mean_std(x)
    """
    return _math.channel_mean_std(_channels(data), runtime=runtime)


def channel_noise_std(data, *, runtime: str | None = None) -> np.ndarray:
    """Robust noise level of every channel: ``median(|x|) / MAD_TO_SIGMA``.

    The usual threshold for spike detection: unlike the standard deviation it is barely moved by
    the spikes themselves. The median here is the magnitude at index ``samples // 2`` when sorted.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    numpy.ndarray
        One value per channel ``(channels,)``.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> noise = dspk.ops.math.channel_noise_std(x)
    """
    return _math.channel_noise_std(_channels(data), runtime=runtime)


def window(name: str, length: int, *, sigma: float | None = None, beta: float | None = None) -> np.ndarray:
    """A symmetric window of ``length`` values.

    Parameters
    ----------
    name : {"hann", "hamming", "gaussian", "kaiser"}
        The window.
    length : int
        Number of values.
    sigma : float, optional
        Standard deviation, in samples; ``"gaussian"`` needs it.
    beta : float, optional
        Shape; ``"kaiser"`` needs it.

    Returns
    -------
    numpy.ndarray
        ``(length,)``.

    Raises
    ------
    ValueError
        For an unknown ``name``, or ``sigma`` / ``beta`` missing from (or given to) a window.

    Examples
    --------
    >>> import dspk
    >>> w = dspk.ops.math.window("kaiser", 51, beta=5.0)
    """
    return _math.window(name, length, sigma=sigma, beta=beta)


__all__ = ["MAD_TO_SIGMA", "channel_mad", "channel_mean_std", "channel_noise_std", "window"]
