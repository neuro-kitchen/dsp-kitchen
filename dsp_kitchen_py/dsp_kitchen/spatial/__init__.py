"""
Spatial operators on ``[channels, samples]``: common average reference, surface Laplacian, and
spatial whitening (ZCA, global or local; fitted on the device, ``epsilon`` required).
"""

from dsp_kitchen_bindings import CommonAverageReference, SpatialWhitening, SurfaceLaplacian, common_average_reference

__all__ = ["CommonAverageReference", "SpatialWhitening", "SurfaceLaplacian", "common_average_reference"]
