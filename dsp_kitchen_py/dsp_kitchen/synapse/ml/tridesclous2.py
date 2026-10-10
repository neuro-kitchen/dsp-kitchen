"""Tridesclous 2 (SpikeInterface's rewrite of Tridesclous by Samuel Garcia and Christophe Pouzat),
ported from its MIT source, running on the GPU.

Bessel band-pass, common median reference and local whitening; locally exclusive detection; local
SVD features; iterative splits with SpikeInterface's isosplit; template cleaning and merging; mean
templates; the Tridesclous peeler (template subtraction in levels, with a matched-filter level);
final merges. Not yet: motion correction (off by default upstream too).

Examples
--------
>>> from dsp_kitchen.synapse.ml import tridesclous2
>>> config = tridesclous2.Config()
>>> result = tridesclous2.run(recording, probe, config)
>>> sorting = result.to_sorting_output(probe)
"""

from __future__ import annotations

from typing import Optional

from dsp_kitchen_bindings import (
    ProbeLayout,
    Recording,
    Tridesclous2Config as Config,
    Tridesclous2Result,
    tridesclous2_provenance as provenance,
)
from dsp_kitchen_bindings import run_tridesclous2 as _run

from ...progress import progress_callback
from .kilosort4 import Progress


def run(
    recording: Recording,
    probe: ProbeLayout,
    config: Config,
    *,
    progress: Progress = True,
    runtime: Optional[str] = None,
) -> Tridesclous2Result:
    """Sorts a whole recording with Tridesclous 2.

    Parameters
    ----------
    recording : Recording
        The recording to sort (channels in the probe's order).
    probe : ProbeLayout
        Contact positions (µm) of the recording's channels.
    config : Config
        Settings (see ``Config``: every setting, with Tridesclous 2's defaults).
    progress : bool or callable, default True
        ``True``: a progress bar per stage; ``False``: none; a callable receives
        ``(stage, step, steps, done, total, unit)``.
    runtime : str, optional
        Compute runtime; default: the current one (``dsp_kitchen.runtime``).

    Returns
    -------
    Tridesclous2Result
        Spikes (sample, unit, scaling), unit templates, noise levels, the preprocessing.
    """
    callback = progress_callback(progress)
    try:
        return _run(recording, probe, config, progress=callback, runtime=runtime)
    finally:
        if hasattr(callback, "close"):
            callback.close()


__all__ = ["Config", "Tridesclous2Result", "provenance", "run"]
