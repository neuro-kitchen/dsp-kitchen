"""
Filter module for dsp-kitchen.
Subdivided into:
  - iir: Infinite Impulse Response filters (Notch, Bandpass)
  - fir: Finite Impulse Response filters
  - non_linear: Non-linear filters (Median, TeagerKaiser)
  - template: Template alignment, scaling, and subtraction
"""

from . import iir, fir, non_linear, template
from .iir import NotchFilter, BandpassFilter, notch_filter, bandpass_filter
from .non_linear import (
    MedianFilter,
    median_filter_9p,
    TeagerKaiser,
    teager_kaiser_filter,
)
from .template import TemplateFilter, subtract_template

__all__ = [
    "iir",
    "fir",
    "non_linear",
    "template",
    "NotchFilter",
    "BandpassFilter",
    "MedianFilter",
    "TeagerKaiser",
    "TemplateFilter",
    "notch_filter",
    "bandpass_filter",
    "median_filter_9p",
    "teager_kaiser_filter",
    "subtract_template",
]
