"""
Mathematical and sample scaling operations for signal processing.
"""

from .._dsp_kitchen import (
    Scale,
    SubtractBaseline,
    Clamp,
    scale_samples,
)

__all__ = [
    "Scale",
    "SubtractBaseline",
    "Clamp",
    "scale_samples",
]
