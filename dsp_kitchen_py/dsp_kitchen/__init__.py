"""
dsp-kitchen: digital signal processing for multi-channel recordings (neural and muscle
electrophysiology), on any GPU or the CPU.

    import dsp_kitchen as dk
    dk.runtime.available()                       # compiled-in runtimes
    y = dk.filter.iir.bandpass_filter(x, 300, 6000, fs=30_000)
    rec = dk.io.Recording("session.ap.bin")

Subpackages: ``filter`` (``iir``, ``fir``, ``non_linear``, ``template``), ``spatial``, ``math``,
``linalg``, ``pipeline``, ``io``, ``synapse`` (``synapse.ml``: sorters and published artifacts),
``runtime``.
"""

from dsp_kitchen_bindings import __version__

from . import filter, io, linalg, math, pipeline, runtime, spatial, synapse

__all__ = ["__version__", "filter", "io", "linalg", "math", "pipeline", "runtime", "spatial", "synapse"]
