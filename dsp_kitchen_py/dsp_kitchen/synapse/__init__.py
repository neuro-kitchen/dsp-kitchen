"""
Neuroscience-specific algorithms: probe layouts, neural spike detection,
spatial deduplication, sub-sample sinc realignment, multi-channel snippet extraction,
3D localization, drift estimation & kriging, clustering (GMM, IsoSplit, Density Peaks),
OMP template deconvolution, HD-EMG cBSS decomposition, out-of-core streaming spike sorting
(`sort_recording`), electrophysiology metrics (ISI violations, SNR, templates with SE,
ACG/CCG, Gaussian firing rates, PSTH, STA, MEP), and `dsp-synapse-ml` deep learning /
Burn-ONNX external sorter bridges (`synapse.ml`, `synapse.onnx`).
"""

from typing import List, Optional, Tuple
from .._bindings import (
    DeduplicatedSpike,
    Kilosort4BasisEmbedder,
    Kilosort4Detector,
    ModelHub,
    MyomatrixBasisEmbedder,
    MyomatrixDetector,
    MyomatrixLatencyAligner,
    MyomatrixSortConfig,
    ProbeLayout,
    SortingOutput,
    SpikeEvent,
    StreamingSortResult,
    WaveformSnippet,
    cluster_density_peaks,
    cluster_gmm,
    cluster_isosplit,
    compare_sortings,
    compare_spike_trains,
    compute_amplitude_cutoff,
    compute_autocorrelogram,
    compute_crosscorrelogram,
    compute_d_prime,
    compute_firing_rate,
    compute_isi,
    compute_isolation_distance,
    compute_presence_ratio,
    compute_psth,
    compute_silhouette_score,
    compute_snr,
    compute_sta,
    compute_template,
    correct_drift_kriging,
    decompose_hdemg_cbss,
    deduplicate_spikes,
    detect_spikes,
    estimate_noise,
    estimate_nonrigid_drift,
    estimate_rigid_drift,
    export_to_phy,
    extract_snippets,
    load_nwb_units,
    load_sorting,
    localize_spikes,
    match_spikes_omp,
    quantify_mep,
    read_kilosort,
    save_nwb_units,
    save_sorting,
    sort_recording,
)
from . import ml


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


def hdemg_4x8_layout(ied_mm: float = 4.0) -> ProbeLayout:
    """Returns a 32-channel (4x8) High-Density Surface EMG grid layout."""
    return ProbeLayout.hdemg_4x8(ied_mm)


def hdemg_8x8_layout(ied_mm: float = 4.0) -> ProbeLayout:
    """Returns a 64-channel (8x8) High-Density Surface EMG grid layout."""
    return ProbeLayout.hdemg_8x8(ied_mm)


def custom_layout(
    name: str,
    positions: List[Tuple[float, float]],
    shank_ids: Optional[List[int]] = None,
) -> ProbeLayout:
    """Creates a custom probe layout from 2D coordinates."""
    return ProbeLayout.from_positions(name, positions, shank_ids)


__all__ = [
    "ml",
    "ModelHub",
    "Kilosort4BasisEmbedder",
    "Kilosort4Detector",
    "MyomatrixSortConfig",
    "MyomatrixBasisEmbedder",
    "MyomatrixDetector",
    "MyomatrixLatencyAligner",
    "ProbeLayout",
    "SpikeEvent",
    "DeduplicatedSpike",
    "WaveformSnippet",
    "StreamingSortResult",
    "SortingOutput",
    "detect_spikes",
    "deduplicate_spikes",
    "estimate_noise",
    "extract_snippets",
    "localize_spikes",
    "estimate_rigid_drift",
    "estimate_nonrigid_drift",
    "correct_drift_kriging",
    "cluster_gmm",
    "cluster_density_peaks",
    "cluster_isosplit",
    "match_spikes_omp",
    "decompose_hdemg_cbss",
    "compare_sortings",
    "compare_spike_trains",
    "save_sorting",
    "load_sorting",
    "export_to_phy",
    "read_kilosort",
    "save_nwb_units",
    "load_nwb_units",
    "compute_isi",
    "compute_snr",
    "compute_template",
    "compute_autocorrelogram",
    "compute_crosscorrelogram",
    "compute_firing_rate",
    "compute_psth",
    "compute_sta",
    "quantify_mep",
    "compute_d_prime",
    "compute_isolation_distance",
    "compute_silhouette_score",
    "compute_amplitude_cutoff",
    "compute_presence_ratio",
    "sort_recording",
    "neuropixels_1_0_layout",
    "neuropixels_2_0_layout",
    "tetrode_layout",
    "utah_array_layout",
    "hdemg_4x8_layout",
    "hdemg_8x8_layout",
    "custom_layout",
]
