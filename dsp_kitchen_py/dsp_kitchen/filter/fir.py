"""
FIR filters: Gaussian smoothing (``scipy.ndimage.gaussian_filter1d``, reflected edges by default).
"""

from dsp_kitchen_bindings import GaussianSmooth, gaussian_smooth

__all__ = ["GaussianSmooth", "gaussian_smooth"]
