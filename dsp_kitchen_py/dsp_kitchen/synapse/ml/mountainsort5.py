"""MountainSort 5 (Chung et al., *Neuron* 2017), ported from its Apache-2.0 source
(``flatironinstitute/mountainsort5``, with ``isosplit6``), running on the GPU.

Scheme 2 (the default) sorts a training stretch (300 s in 10 s chunks): locally exclusive
detection, masked snippets, PCA, the isosplit6 subdivision method, median templates and their
alignment. It then trains one classifier per channel on those units and classifies every spike of
the recording. Scheme 1 is the first part on the whole recording. Preprocessing follows
SpikeInterface's wrapper: band-pass 300–6000 Hz, then global whitening.

Examples
--------
>>> from dsp_kitchen.synapse.ml import mountainsort5
>>> config = mountainsort5.Config()
>>> result = mountainsort5.run(recording, probe, config)
>>> sorting = result.to_sorting_output(probe)
"""

from __future__ import annotations

from typing import Optional

from dsp_kitchen_bindings import (
    Mountainsort5Config as Config,
    Mountainsort5Result,
    ProbeLayout,
    Recording,
    mountainsort5_provenance as provenance,
)
from dsp_kitchen_bindings import run_mountainsort5 as _run

from ...progress import progress_callback
from .kilosort4 import Progress


def run(
    recording: Recording,
    probe: ProbeLayout,
    config: Config,
    *,
    progress: Progress = True,
    runtime: Optional[str] = None,
) -> Mountainsort5Result:
    """Sorts a whole recording with MountainSort 5.

    Parameters
    ----------
    recording : Recording
        The recording to sort (channels in the probe's order).
    probe : ProbeLayout
        Contact positions (µm) of the recording's channels: the detection and snippet radii are
        distances on it.
    config : Config
        Settings (see ``Config``: every setting, with MountainSort 5's defaults). Snippet lengths
        count samples.
    progress : bool or callable, default True
        ``True``: a progress bar per stage; ``False``: none; a callable receives
        ``(stage, step, steps, done, total, unit)``.
    runtime : str, optional
        Compute runtime; default: the current one (``dsp_kitchen.runtime``).

    Returns
    -------
    Mountainsort5Result
        Spikes (sample, unit, amplitude, detection channel), unit templates, the preprocessing.
    """
    callback = progress_callback(progress)
    try:
        return _run(recording, probe, config, progress=callback, runtime=runtime)
    finally:
        if hasattr(callback, "close"):
            callback.close()


__all__ = ["Config", "Mountainsort5Result", "provenance", "run"]
