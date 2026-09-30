"""
dsp-kitchen: High-throughput real-time digital signal processing library in Rust for large-scale electrophysiology and neural time-series.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import os
import sys
import numpy as np

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from . import _bindings as _dsp_kitchen
from ._bindings import (
    PCA,
    BandpassFilter,
    BandstopFilter,
    Clamp,
    CommonAverageReference,
    ContrastiveWaveformEmbedder,
    DartsortVaeEmbedder,
    DeduplicatedSpike,
    DspSession,
    HighpassFilter,
    LowpassFilter,
    MedianFilter,
    MmapRecording,
    NotchFilter,
    NwbZarrRecording,
    OnnxModelRunner,
    Pipeline,
    ProbeLayout,
    Scale,
    SingleChannelDenoiser,
    SpatiotemporalUnetDenoiser,
    SpikeEvent,
    StreamingSortResult,
    SubtractBaseline,
    TeagerKaiser,
    TemplateFilter,
    UnitQualityClassifier,
    WaveformSnippet,
    __version__,
    bandpass_filter,
    common_average_reference,
    compute_isi,
    compute_snr,
    compute_template,
    deduplicate_spikes,
    detect_spikes,
    estimate_noise,
    extract_snippets,
    highpass_filter,
    list_nwb_series,
    lowpass_filter,
    median_filter_9p,
    notch_filter,
    scale_samples,
    sort_recording,
    subtract_template,
    teager_kaiser_filter,
)

# Import submodules
from . import filter, io, linalg, math, pipeline, spatial, synapse

# Backward-compatible plural alias (`dsp_kitchen.filters`) without duplicating directory files
filters = filter
sys.modules[f"{__name__}.filters"] = filter

# Convenience probe aliases
neuropixels_1_0_layout = synapse.neuropixels_1_0_layout
neuropixels_2_0_layout = synapse.neuropixels_2_0_layout
tetrode_layout = synapse.tetrode_layout
utah_array_layout = synapse.utah_array_layout

__all__ = [
    # Submodules
    "filter",
    "filters",
    "io",
    "spatial",
    "math",
    "pipeline",
    "linalg",
    "synapse",
    "_dsp_kitchen",
    # Core Pipeline & Types
    "Pipeline",
    "ProbeLayout",
    "SpikeEvent",
    "DeduplicatedSpike",
    "WaveformSnippet",
    "StreamingSortResult",
    "MmapRecording",
    "NwbZarrRecording",
    "DspSession",
    "PCA",
    # Deep Learning & Burn-ONNX
    "SpatiotemporalUnetDenoiser",
    "SingleChannelDenoiser",
    "DartsortVaeEmbedder",
    "ContrastiveWaveformEmbedder",
    "UnitQualityClassifier",
    "OnnxModelRunner",
    # Stage Classes
    "Scale",
    "SubtractBaseline",
    "Clamp",
    "NotchFilter",
    "BandpassFilter",
    "HighpassFilter",
    "LowpassFilter",
    "BandstopFilter",
    "CommonAverageReference",
    "MedianFilter",
    "TeagerKaiser",
    "TemplateFilter",
    # Direct Functions
    "list_nwb_series",
    "sort_recording",
    "notch_filter",
    "bandpass_filter",
    "highpass_filter",
    "lowpass_filter",
    "common_average_reference",
    "scale_samples",
    "median_filter_9p",
    "teager_kaiser_filter",
    "subtract_template",
    "detect_spikes",
    "deduplicate_spikes",
    "estimate_noise",
    "extract_snippets",
    "compute_isi",
    "compute_snr",
    "compute_template",
    # Helpers
    "get_local_path",
    "resolve_data_path",
    "load_recording",
    "open_nwb_zarr",
    "neuropixels_1_0_layout",
    "neuropixels_2_0_layout",
    "tetrode_layout",
    "utah_array_layout",
    "__version__",
]


def get_local_path() -> Path:
    """
    Returns the base path defined by LOCAL_PATH in .env or environment variables.
    """
    val = os.getenv("LOCAL_PATH")
    if val and val.strip():
        return Path(val.strip().strip('"').strip("'"))

    return Path(__file__).resolve().parent.parent.parent.parent


def resolve_data_path(rel_or_abs_path: Union[str, Path]) -> Path:
    """
    Resolves a file path: if relative, prepends LOCAL_PATH.
    """
    p = Path(rel_or_abs_path)
    if p.is_absolute():
        return p
    return get_local_path() / p


def open_nwb_zarr(
    path: Union[str, Path],
    series: Optional[str] = None,
) -> NwbZarrRecording:
    """
    Opens an NWB Zarr v3 store (`.nwb.zarr`) or general `dsp-io` recording.
    If `series` is omitted, opens the largest `ElectricalSeries` in `/acquisition`.
    """
    resolved_path = resolve_data_path(path)
    return NwbZarrRecording(str(resolved_path), series=series)


def load_recording(
    path: Union[str, Path] = "playground/data/mock_signal_384ch.bin",
    channels: int = 384,
    samples: int = 0,
    sample_rate: float = 30000.0,
) -> Tuple[MmapRecording, np.ndarray]:
    """
    Opens a raw binary electrophysiology recording via zero-copy memory mapping.
    If the path is relative, it is automatically resolved against LOCAL_PATH.

    Returns:
        tuple of (MmapRecording, np.ndarray) where the ndarray is a zero-copy 2D view
        of shape (channels, samples).
    """
    resolved_path = resolve_data_path(path)
    rec = MmapRecording(
        str(resolved_path),
        channels=channels,
        samples=samples,
        sample_rate=sample_rate,
    )
    arr = rec.to_numpy()
    return rec, arr
