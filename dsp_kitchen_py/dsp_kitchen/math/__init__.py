"""
Mathematical and sample scaling operations for signal processing.
"""

from .._bindings import (
    Clamp,
    Scale,
    SubtractBaseline,
    scale_samples,
)

__all__ = [
    "Scale",
    "SubtractBaseline",
    "Clamp",
    "scale_samples",
]
