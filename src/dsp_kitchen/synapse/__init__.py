"""
Neuroscience-specific algorithms: probe layouts, neural spike detection, waveform extraction,
and Quiroga noise estimation.
"""

from .._dsp_kitchen import (
    ProbeLayout,
    SpikeEvent,
    detect_spikes,
    estimate_noise,
)

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

__all__ = [
    "ProbeLayout",
    "SpikeEvent",
    "detect_spikes",
    "estimate_noise",
    "neuropixels_1_0_layout",
    "neuropixels_2_0_layout",
    "tetrode_layout",
    "utah_array_layout",
]
