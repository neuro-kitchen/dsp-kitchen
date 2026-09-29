"""
dsp-kitchen: High-throughput real-time digital signal processing library in Rust for large-scale electrophysiology and neural time-series.
"""

from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path
import os
import numpy as np
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# Import the native PyO3 Rust extension module
from ._dsp_kitchen import (
    ProbeLayout,
    SpikeEvent,
    MmapRecording,
    DspSession,
    Pipeline,
    Scale,
    SubtractBaseline,
    Clamp,
    NotchFilter,
    BandpassFilter,
    CommonAverageReference,
    MedianFilter,
    TeagerKaiser,
    PCA,
    notch_filter,
    bandpass_filter,
    common_average_reference,
    scale_samples,
    median_filter_9p,
    teager_kaiser_filter,
    detect_spikes,
    estimate_noise,
    __version__,
)

# Import submodules
from . import filter
from . import filters
from . import spatial
from . import math
from . import pipeline
from . import linalg
from . import synapse

# Convenience probe aliases
neuropixels_1_0_layout = synapse.neuropixels_1_0_layout
neuropixels_2_0_layout = synapse.neuropixels_2_0_layout
tetrode_layout = synapse.tetrode_layout
utah_array_layout = synapse.utah_array_layout

__all__ = [
    # Submodules
    "filter",
    "filters",
    "spatial",
    "math",
    "pipeline",
    "linalg",
    "synapse",
    # Core Pipeline & Types
    "Pipeline",
    "ProbeLayout",
    "SpikeEvent",
    "MmapRecording",
    "DspSession",
    "PCA",
    # Stage Classes
    "Scale",
    "SubtractBaseline",
    "Clamp",
    "NotchFilter",
    "BandpassFilter",
    "CommonAverageReference",
    "MedianFilter",
    "TeagerKaiser",
    # Direct Functions
    "notch_filter",
    "bandpass_filter",
    "common_average_reference",
    "scale_samples",
    "median_filter_9p",
    "teager_kaiser_filter",
    "detect_spikes",
    "estimate_noise",
    # Helpers
    "get_local_path",
    "resolve_data_path",
    "load_recording",
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

    return Path(__file__).resolve().parent.parent.parent

def resolve_data_path(rel_or_abs_path: Union[str, Path]) -> Path:
    """
    Resolves a file path: if relative, prepends LOCAL_PATH.
    """
    p = Path(rel_or_abs_path)
    if p.is_absolute():
        return p
    return get_local_path() / p

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
    rec = MmapRecording(str(resolved_path), channels=channels, samples=samples, sample_rate=sample_rate)
    arr = rec.to_numpy()
    return rec, arr
