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
)
from dsp_kitchen_bindings import run as _run

from ...progress import progress_callback


def run(recording, probe, config, *, templates=None, preprocessing_from=None, progress=True, runtime=None):
    """Kilosort4 over the whole ``recording`` (preprocessing fit, universal templates,
    detection), showing a progress bar per stage with the time left (``progress=False``: none; a
    callable receives ``(stage, step, steps, done, total, unit)``). ``templates``: predefined
    universal templates when ``config.templates_from_data`` is off. ``preprocessing_from``: an
    earlier result on the same recording with the same fit settings (skips the fit)."""
    callback = progress_callback(progress)
    try:
        return _run(
            recording,
            probe,
            config,
            templates=templates,
            preprocessing_from=preprocessing_from,
            progress=callback,
            runtime=runtime,
        )
    finally:
        if hasattr(callback, "close"):
            callback.close()

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
