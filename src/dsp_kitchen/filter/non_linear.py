"""
Non-linear filters (e.g. median filtering via sorting networks) for neural signal processing.
"""

from .._dsp_kitchen import (
    MedianFilter,
    median_filter_9p,
)

__all__ = [
    "MedianFilter",
    "median_filter_9p",
]
