"""
Spatial processing algorithms for multi-channel neural recordings.
"""

from .._bindings import (
    CommonAverageReference,
    SpatialWhitening,
    SurfaceLaplacian,
    common_average_reference,
)

__all__ = [
    "CommonAverageReference",
    "SpatialWhitening",
    "SurfaceLaplacian",
    "common_average_reference",
]
