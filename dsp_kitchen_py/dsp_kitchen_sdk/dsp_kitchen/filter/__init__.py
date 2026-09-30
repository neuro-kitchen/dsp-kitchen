"""
Filter module for dsp-kitchen.
Subdivided into:
  - iir: Infinite Impulse Response filters (Butterworth band/high/low/stop, Notch)
  - fir: Finite Impulse Response filters
  - non_linear: Non-linear filters (Median, TeagerKaiser)
  - template: Template alignment, scaling, and subtraction
"""

from . import fir, iir, non_linear, template
from .iir import (
    BandpassFilter,
    BandstopFilter,
    HighpassFilter,
    LowpassFilter,
    NotchFilter,
    bandpass_filter,
    highpass_filter,
    lowpass_filter,
    notch_filter,
)
from .non_linear import (
    MedianFilter,
    TeagerKaiser,
    median_filter_9p,
    teager_kaiser_filter,
)
from .template import TemplateFilter, subtract_template

__all__ = [
    "fir",
    "iir",
    "non_linear",
    "template",
    "NotchFilter",
    "BandpassFilter",
    "HighpassFilter",
    "LowpassFilter",
    "BandstopFilter",
    "MedianFilter",
    "TeagerKaiser",
    "TemplateFilter",
    "notch_filter",
    "bandpass_filter",
    "highpass_filter",
    "lowpass_filter",
    "median_filter_9p",
    "teager_kaiser_filter",
    "subtract_template",
]
