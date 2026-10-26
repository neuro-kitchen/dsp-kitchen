"""
PCA, probabilistic PCA and FastICA fitted on the device. Data is ``[channels, samples]``
(channels are features); defaults follow scikit-learn.
"""

from dsp_kitchen_bindings import PCA, PPCA, FastICA

__all__ = ["PCA", "PPCA", "FastICA"]
