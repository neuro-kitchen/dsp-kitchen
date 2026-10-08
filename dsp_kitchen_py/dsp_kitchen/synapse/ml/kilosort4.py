"""Kilosort4 (Pachitariu et al., *Nature Methods* 2024), written from the paper and its published
defaults, running on the GPU.

``run(recording, probe, Config())`` sorts a whole recording: preprocessing fitted on it (high-pass,
common average reference, local whitening), universal templates learned from it, universal-template
detection, a first clustering into units, learned templates, learned-template matching, and the
clustering of the matched spikes into the final units. The stages are also available one by one
(``create_preprocessing``, ``extract_clips``, ``learn_universal_templates``, ``TemplateCentres``,
``detect_universal``). Not yet: the refractory criteria, the final merges, drift correction.

Examples
--------
>>> from dsp_kitchen.io import Recording
>>> import dsp_kitchen.synapse as syn
>>> from dsp_kitchen.synapse.ml import kilosort4
>>> recording = Recording("recording.bin")
>>> probe = syn.ProbeLayout.from_recording("recording.bin")
>>> result = kilosort4.run(recording, probe, kilosort4.Config(th_learned=7.0))
>>> sorting = result.to_sorting_output(probe)
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from dsp_kitchen_bindings import (
    EmusortConfig,
    Kilosort4Config as Config,
    Kilosort4Result,
    ProbeLayout,
    Recording,
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

Progress = Union[bool, Callable[[str, int, int, int, int, str], None]]


def run(
    recording: Recording,
    probe: ProbeLayout,
    config: Union[Config, EmusortConfig],
    *,
    templates: Optional[UniversalTemplates] = None,
    preprocessing_from: Optional[Kilosort4Result] = None,
    progress: Progress = True,
    runtime: Optional[str] = None,
) -> Kilosort4Result:
    """Sorts a whole recording with Kilosort4.

    The recording is streamed in batches of ``config.batch_size`` samples (memory stays bounded
    whatever its length); every stage runs on the device.

    Parameters
    ----------
    recording : Recording
        The recording to sort (any format ``Recording`` opens; channels in the probe's order).
    probe : ProbeLayout
        Contact positions (µm) of the recording's channels.
    config : Config or EmusortConfig
        Settings (see ``Config``); an ``EmusortConfig`` runs Kilosort4's stages with EMUsort's
        settings (prefer ``emusort.run``).
    templates : UniversalTemplates, optional
        Predefined universal templates, used when ``config.templates_from_data`` is ``False``
        (``UniversalTemplates.from_npz`` / ``from_hub``); without them the hub's are fetched.
    preprocessing_from : Kilosort4Result, optional
        An earlier result on the same recording with the same fit settings: its preprocessing is
        reused (no second fit pass).
    progress : bool or callable, default True
        ``True``: a progress bar per stage with the time left; ``False``: none; a callable receives
        ``(stage, step, steps, done, total, unit)`` for each report.
    runtime : str, optional
        Compute runtime (``"wgpu"``, ``"cuda"``, ``"cpu"``, …); default: the current one
        (``dsp_kitchen.runtime``).

    Returns
    -------
    Kilosort4Result
        Spikes (``result.spikes()``), units (``result.n_units``, ``result.to_sorting_output(probe)``),
        templates and the fitted preprocessing (``result.preprocessing``).
    """
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
