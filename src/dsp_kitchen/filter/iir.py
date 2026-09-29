"""
Infinite Impulse Response (IIR) digital filters for neural signal processing.
"""

from .._dsp_kitchen import (
    NotchFilter,
    BandpassFilter,
    notch_filter,
    bandpass_filter,
)

__all__ = [
    "NotchFilter",
    "BandpassFilter",
    "notch_filter",
    "bandpass_filter",
]
