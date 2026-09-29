"""
dsp-kitchen: High-throughput real-time digital signal processing library in Rust for large-scale neuroscience electrophysiology.
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
    MmapRecording,
    DspSession,
    Pipeline,
    Scale,
    SubtractBaseline,
    NotchFilter,
    BandpassFilter,
    CommonAverageReference,
    MedianFilter,
    notch_filter,
    bandpass_filter,
    common_average_reference,
    scale_samples,
    median_filter_9p,
    __version__,
)

# Import submodules
from . import filter
from . import filters
from . import spatial
from . import math
from . import pipeline

__all__ = [
    # Submodules
    "filter",
    "filters",
    "spatial",
    "math",
    "pipeline",
    # Core Pipeline & Types
    "Pipeline",
    "ProbeLayout",
    "MmapRecording",
    "DspSession",
    # Stage Classes
    "Scale",
    "SubtractBaseline",
    "NotchFilter",
    "BandpassFilter",
    "CommonAverageReference",
    "MedianFilter",
    # Direct Functions
    "notch_filter",
    "bandpass_filter",
    "common_average_reference",
    "scale_samples",
    "median_filter_9p",
    # Helpers
    "get_local_path",
    "resolve_data_path",
    "load_recording",
    "neuropixels_1_0_layout",
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

def neuropixels_1_0_layout() -> ProbeLayout:
    """
    Returns the standard 384-channel Neuropixels 1.0 probe layout.
    """
    return ProbeLayout.neuropixels_1_0()
