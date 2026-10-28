"""Per-channel statistics on the device, and window designs."""

from __future__ import annotations

import numpy as np

from dspk._native import ops as _ops
from dspk.ops.filter._common import _channels

_math = _ops.math

MAD_TO_SIGMA: float = _math.MAD_TO_SIGMA


def channel_mean_std(data, *, runtime: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation of every channel of ``data`` ``(channels, samples)``."""
    return _math.channel_mean_std(_channels(data), runtime=runtime)


def channel_noise_std(data, *, runtime: str | None = None) -> np.ndarray:
    """Robust noise level of every channel: ``median(|x|) / MAD_TO_SIGMA`` (the magnitude at index
    ``samples // 2`` when sorted)."""
    return _math.channel_noise_std(_channels(data), runtime=runtime)


def window(name: str, length: int, *, sigma: float | None = None, beta: float | None = None) -> np.ndarray:
    """Window of ``length`` values: ``"hann"``, ``"hamming"``, ``"gaussian"`` (``sigma``) or
    ``"kaiser"`` (``beta``), symmetric."""
    return _math.window(name, length, sigma=sigma, beta=beta)


__all__ = ["MAD_TO_SIGMA", "channel_mean_std", "channel_noise_std", "window"]
