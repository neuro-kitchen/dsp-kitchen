"""Median and Teager–Kaiser on the device."""

from __future__ import annotations

import numpy as np

from dspk.ops.filter._common import _channels, _filter

MEDIAN_DEFAULT_EDGE: str = _filter.MEDIAN_DEFAULT_EDGE
TEAGER_KAISER_DEFAULT_EDGE: str = _filter.TEAGER_KAISER_DEFAULT_EDGE
MAX_MEDIAN_WIDTH: int = _filter.MAX_MEDIAN_WIDTH


def median(data, width: int, *, edge: str = MEDIAN_DEFAULT_EDGE, runtime: str | None = None) -> np.ndarray:
    """Running median over an odd ``width`` (at most :data:`MAX_MEDIAN_WIDTH`) along time; with the
    default ``edge="zeros"`` this is ``scipy.signal.medfilt``."""
    return _filter.median(_channels(data), width, edge=edge, runtime=runtime)


def teager_kaiser(data, *, edge: str = TEAGER_KAISER_DEFAULT_EDGE, runtime: str | None = None) -> np.ndarray:
    """Teager–Kaiser energy ``x[t]**2 - x[t-1] * x[t+1]`` along time; the neighbours of the end
    samples come from ``edge``."""
    return _filter.teager_kaiser(_channels(data), edge=edge, runtime=runtime)


__all__ = ["MAX_MEDIAN_WIDTH", "MEDIAN_DEFAULT_EDGE", "TEAGER_KAISER_DEFAULT_EDGE", "median", "teager_kaiser"]
