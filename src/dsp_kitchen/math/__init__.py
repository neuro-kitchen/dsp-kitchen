"""
Mathematical and sample scaling operations for neural signal processing.
"""

from .._dsp_kitchen import (
    Scale,
    SubtractBaseline,
    scale_samples,
)

__all__ = [
    "Scale",
    "SubtractBaseline",
    "scale_samples",
]
