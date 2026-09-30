"""
Non-linear filters (e.g. median filtering via sorting networks, Teager-Kaiser Energy Operator)
for neural signal processing.
"""

from .._bindings import (
    MedianFilter,
    TeagerKaiser,
    median_filter_9p,
    teager_kaiser_filter,
)

__all__ = [
    "MedianFilter",
    "median_filter_9p",
    "TeagerKaiser",
    "teager_kaiser_filter",
]
