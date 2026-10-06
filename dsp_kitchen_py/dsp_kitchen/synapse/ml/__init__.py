"""
Sorters reimplemented from their papers, each with its provenance (``kilosort4``, ``emusort``),
and the catalog of published artifacts (``ModelHub``, when built with the ``hub`` feature).
Cite the authors: ``kilosort4.provenance().citation()``.
"""

import dsp_kitchen_bindings as _native

from . import emusort, kilosort4

__all__ = ["emusort", "kilosort4"]

if hasattr(_native, "ModelHub"):
    ModelHub = _native.ModelHub
    __all__.append("ModelHub")
