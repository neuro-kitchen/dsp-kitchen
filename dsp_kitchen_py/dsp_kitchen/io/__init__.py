"""
Recordings: ``Recording`` (any format dsp-io reads, lazily sliced; ``list_sources`` lists a
file's signals), ``MmapRecording`` (raw binary with zero-copy NumPy views) and
``SyntheticRecording`` (procedural, with ground-truth spike times).
"""

from dsp_kitchen_bindings import MmapRecording, Recording, SyntheticRecording, list_sources

__all__ = ["MmapRecording", "Recording", "SyntheticRecording", "list_sources"]
