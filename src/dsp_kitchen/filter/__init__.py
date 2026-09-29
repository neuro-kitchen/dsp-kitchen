"""
Filter module for dsp-kitchen.
Subdivided into:
  - iir: Infinite Impulse Response filters (Notch, Bandpass)
  - fir: Finite Impulse Response filters
  - non_linear: Non-linear filters (Median)
"""

from . import iir, fir, non_linear
from .iir import NotchFilter, BandpassFilter, notch_filter, bandpass_filter
from .non_linear import MedianFilter, median_filter_9p

__all__ = [
    "iir",
    "fir",
    "non_linear",
    "NotchFilter",
    "BandpassFilter",
    "MedianFilter",
    "notch_filter",
    "bandpass_filter",
    "median_filter_9p",
]
