"""
Deep learning spike-sorting models (`dsp-synapse-ml`): waveform denoisers,
variational & contrastive latent embedders, and automated unit curation.
"""

from .._bindings import (
    ContrastiveWaveformEmbedder,
    DartsortVaeEmbedder,
    SingleChannelDenoiser,
    SpatiotemporalUnetDenoiser,
    UnitQualityClassifier,
)

__all__ = [
    "SpatiotemporalUnetDenoiser",
    "SingleChannelDenoiser",
    "DartsortVaeEmbedder",
    "ContrastiveWaveformEmbedder",
    "UnitQualityClassifier",
]
