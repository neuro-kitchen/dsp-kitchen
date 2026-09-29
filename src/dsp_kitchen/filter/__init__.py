"""
Filter module for dsp-kitchen.
Subdivided into:
  - iir: Infinite Impulse Response filters (Notch, Bandpass)
  - fir: Finite Impulse Response filters
  - non_linear: Non-linear filters (Median, TeagerKaiser)
"""

from . import iir, fir, non_linear
from .iir import NotchFilter, BandpassFilter, notch_filter, bandpass_filter
from .non_linear import (
    MedianFilter,
    median_filter_9p,
    TeagerKaiser,
    teager_kaiser_filter,
)

__all__ = [
    "iir",
    "fir",
    "non_linear",
    "NotchFilter",
    "BandpassFilter",
    "MedianFilter",
    "TeagerKaiser",
    "notch_filter",
    "bandpass_filter",
    "median_filter_9p",
    "teager_kaiser_filter",
]
