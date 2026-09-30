"""
Neuroscience-specific algorithms: probe layouts, neural spike detection,
spatial deduplication, sub-sample sinc realignment, multi-channel snippet extraction,
electrophysiology metrics (ISI violations, SNR, templates), and `dsp-synapse-ml`
deep learning / Burn-ONNX external sorter bridges (`synapse.ml`, `synapse.onnx`).
"""

from typing import List, Optional, Tuple
from .._bindings import (
    ContrastiveWaveformEmbedder,
    DartsortVaeEmbedder,
    DeduplicatedSpike,
    OnnxModelRunner,
    ProbeLayout,
    SingleChannelDenoiser,
    SpatiotemporalUnetDenoiser,
    SpikeEvent,
    UnitQualityClassifier,
    WaveformSnippet,
    compute_isi,
    compute_snr,
    compute_template,
    deduplicate_spikes,
    detect_spikes,
    estimate_noise,
    extract_snippets,
)
from . import ml, onnx


def neuropixels_1_0_layout() -> ProbeLayout:
    """Returns the standard 384-channel Neuropixels 1.0 probe layout."""
    return ProbeLayout.neuropixels_1_0()


def neuropixels_2_0_layout() -> ProbeLayout:
    """Returns the standard Neuropixels 2.0 (4-shank) probe layout."""
    return ProbeLayout.neuropixels_2_0()


def tetrode_layout() -> ProbeLayout:
    """Returns the standard 4-channel tetrode layout."""
    return ProbeLayout.tetrode()


def utah_array_layout() -> ProbeLayout:
    """Returns the standard 10x10 (96-channel) Utah array layout."""
    return ProbeLayout.utah_array()


def custom_layout(
    name: str,
    positions: List[Tuple[float, float]],
    shank_ids: Optional[List[int]] = None,
) -> ProbeLayout:
    """Creates a custom probe layout from 2D coordinates."""
    return ProbeLayout.from_positions(name, positions, shank_ids)


__all__ = [
    "ml",
    "onnx",
    "ProbeLayout",
    "SpikeEvent",
    "DeduplicatedSpike",
    "WaveformSnippet",
    "SpatiotemporalUnetDenoiser",
    "SingleChannelDenoiser",
    "DartsortVaeEmbedder",
    "ContrastiveWaveformEmbedder",
    "UnitQualityClassifier",
    "OnnxModelRunner",
    "detect_spikes",
    "deduplicate_spikes",
    "estimate_noise",
    "extract_snippets",
    "compute_isi",
    "compute_snr",
    "compute_template",
    "neuropixels_1_0_layout",
    "neuropixels_2_0_layout",
    "tetrode_layout",
    "utah_array_layout",
    "custom_layout",
]
