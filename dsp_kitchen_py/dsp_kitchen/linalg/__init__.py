"""
Linear algebra and dimensionality reduction primitives.
"""

from .._bindings import FastICA, PCA, PPCA

__all__ = [
    "PCA",
    "PPCA",
    "FastICA",
]
