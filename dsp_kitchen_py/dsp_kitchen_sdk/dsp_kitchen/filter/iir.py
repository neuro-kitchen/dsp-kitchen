"""
Infinite Impulse Response (IIR) digital filters for neural signal processing.
"""

from .._bindings import (
    BandpassFilter,
    NotchFilter,
    bandpass_filter,
    notch_filter,
)

__all__ = [
    "NotchFilter",
    "BandpassFilter",
    "notch_filter",
    "bandpass_filter",
]
