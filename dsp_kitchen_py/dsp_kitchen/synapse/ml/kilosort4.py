"""
Kilosort4 (Pachitariu et al., Nature Methods 2024), written from the paper and its published
defaults. ``run(recording, probe, Config())`` runs the implemented stages over a whole recording in
Rust (halo windows of ``batch_size``): preprocessing and whitening, universal templates learned
from the data, universal-template detection. The stages are also available one by one.
Clustering, deconvolution and merging are not implemented yet.
"""

from dsp_kitchen_bindings import (
    Kilosort4Config as Config,
    Kilosort4Result,
    TemplateCentres,
    UniversalTemplates,
    create_preprocessing,
    detect_universal,
    extract_clips,
    kilosort4_provenance as provenance,
    learn_universal_templates,
    run,
)

__all__ = [
    "Config",
    "Kilosort4Result",
    "run",
    "create_preprocessing",
    "TemplateCentres",
    "UniversalTemplates",
    "detect_universal",
    "extract_clips",
    "learn_universal_templates",
    "provenance",
]
