"""
Deep learning spike-sorting models (`dsp-synapse-ml`): Pretrained Model Hub (`ModelHub`),
waveform denoisers, variational & contrastive latent embedders, and automated unit curation.
"""

from .._bindings import (
    EmusortBasisEmbedder,
    EmusortDetector,
    EmusortLatencyAligner,
    EmusortSortConfig,
    Kilosort4BasisEmbedder,
    Kilosort4Detector,
    ModelHub,
    MyomatrixBasisEmbedder,
    MyomatrixDetector,
    MyomatrixLatencyAligner,
    MyomatrixSortConfig,
)

# Canonical EMUsort uppercase aliases
EMUsortBasisEmbedder = EmusortBasisEmbedder
EMUsortDetector = EmusortDetector
EMUsortLatencyAligner = EmusortLatencyAligner
EMUsortSortConfig = EmusortSortConfig

__all__ = [
    "ModelHub",
    "Kilosort4BasisEmbedder",
    "Kilosort4Detector",
    "EmusortSortConfig",
    "EmusortBasisEmbedder",
    "EmusortDetector",
    "EmusortLatencyAligner",
    "EMUsortSortConfig",
    "EMUsortBasisEmbedder",
    "EMUsortDetector",
    "EMUsortLatencyAligner",
    "MyomatrixSortConfig",
    "MyomatrixBasisEmbedder",
    "MyomatrixDetector",
    "MyomatrixLatencyAligner",
]


