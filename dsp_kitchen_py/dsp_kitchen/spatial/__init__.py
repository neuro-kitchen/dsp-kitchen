"""
Spatial operators on ``[channels, samples]``: common average and common median reference, surface
Laplacian, and
spatial whitening (ZCA, global or local; fitted on the device, ``epsilon`` required).
"""

from dsp_kitchen_bindings import (
    CommonAverageReference,
    CommonMedianReference,
    SpatialWhitening,
    SurfaceLaplacian,
    common_average_reference,
    common_median_reference,
)

__all__ = [
    "CommonAverageReference",
    "CommonMedianReference",
    "SpatialWhitening",
    "SurfaceLaplacian",
    "common_average_reference",
    "common_median_reference",
]
