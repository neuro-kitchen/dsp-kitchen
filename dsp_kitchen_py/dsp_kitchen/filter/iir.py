"""
Infinite Impulse Response (IIR) digital filters for neural signal processing.

Butterworth filters of any order and notch filters, run forward (causal) or forward-backward
(zero phase, the default).
"""

from .._bindings import (
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

__all__ = [
    "NotchFilter",
    "BandpassFilter",
    "HighpassFilter",
    "LowpassFilter",
    "BandstopFilter",
    "notch_filter",
    "bandpass_filter",
    "highpass_filter",
    "lowpass_filter",
]
