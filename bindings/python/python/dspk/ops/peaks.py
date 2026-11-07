"""Peak candidates on the device."""

from __future__ import annotations

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_peaks = _ops.peaks


def candidates(data, heights, *, polarity: str = "positive", runtime: str | None = None) -> dict[str, np.ndarray]:
    """Local extrema of every channel that reach the channel's height.

    Strict on the left (``x[t-1] < x[t] >= x[t+1]``): a flat peak is reported at its first
    sample; the first and last samples are never peaks.

    Parameters
    ----------
    data : array_like
        Signal ``(channels, samples)``. float32, int8, int16, uint16 and int32 arrays go to the
        device as they are and are converted there; other dtypes and lists become float32 first.
    heights : float or array_like
        Height per channel ``(channels,)``, or one value for all.
    polarity : {"positive", "negative", "both"}
        ``"positive"``: maxima ``>= height``. ``"negative"``: minima ``<= -height``.
        ``"both"``: either.
    runtime : str, optional
        A runtime from :func:`dspk.runtime.available` (``"cuda"``, ``"hip"``, ``"wgpu"``,
        ``"cpu"``) for this call. By default: the one chosen with :func:`dspk.runtime.set`, else
        ``DSPK_RUNTIME``, else the first available (GPUs first).

    Returns
    -------
    dict of numpy.ndarray
        ``"channel"``, ``"sample"`` and ``"value"`` of every peak, ordered by channel then sample.

    Raises
    ------
    ValueError
        For an unknown ``polarity``, ``heights`` of another length than the channels, or ``data``
        not 2-D.
    RuntimeError
        When the runtime has no device.

    Examples
    --------
    >>> import numpy as np
    >>> import dspk
    >>> x = np.random.default_rng(0).standard_normal((4, 30_000)).astype(np.float32)
    >>> noise = dspk.ops.math.channel_noise_std(x)
    >>> peaks = dspk.ops.peaks.candidates(x, 4 * noise, polarity="negative")
    """
    x = _channels(data)
    h = np.broadcast_to(np.asarray(heights, dtype=np.float32), (x.shape[0],)).copy()
    channel, sample, value = _peaks.candidates(x, h, polarity=polarity, runtime=runtime)
    return {"channel": channel, "sample": sample, "value": value}


__all__ = ["candidates"]
