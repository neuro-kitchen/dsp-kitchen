"""Peak candidates on the device."""

from __future__ import annotations

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_peaks = _ops.peaks


def candidates(data, heights, *, polarity: str = "positive", runtime: str | None = None) -> dict[str, np.ndarray]:
    """Local extrema of ``data`` ``(channels, samples)`` reaching each channel's height.

    ``heights``: one value per channel (a scalar is used for all). ``polarity``: ``"positive"``
    (local maxima ``>= height``), ``"negative"`` (local minima ``<= -height``) or ``"both"`` (either:
    maxima reaching ``height`` and minima reaching ``-height``).
    Strict on the left (``x[t-1] < x[t] >= x[t+1]``): a flat peak reports its first sample; the first
    and last samples are never peaks. Returns ``{"channel", "sample", "value"}`` arrays, ordered by
    channel then sample.
    """
    x = _channels(data)
    h = np.broadcast_to(np.asarray(heights, dtype=np.float32), (x.shape[0],)).copy()
    channel, sample, value = _peaks.candidates(x, h, polarity=polarity, runtime=runtime)
    return {"channel": channel, "sample": sample, "value": value}


__all__ = ["candidates"]
