"""
Spatial processing algorithms for multi-channel neural recordings.
"""

from .._bindings import (
    CommonAverageReference,
    common_average_reference,
)

__all__ = [
    "CommonAverageReference",
    "common_average_reference",
]
