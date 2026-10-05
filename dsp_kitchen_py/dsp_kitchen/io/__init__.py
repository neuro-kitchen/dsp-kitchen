"""
Zero-copy recording I/O (`MmapRecording`), NWB Zarr v3 / `dsp-io` reader (`NwbZarrRecording`),
and series discovery utilities.
"""

from .._bindings import MmapRecording, NwbZarrRecording, list_nwb_series

__all__ = [
    "MmapRecording",
    "NwbZarrRecording",
    "list_nwb_series",
]
