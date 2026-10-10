"""SpyKING CIRCUS 2 (SpikeInterface's sorter, after Yger et al., *eLife* 2018), ported from its MIT
source, running on the GPU.

Bessel band-pass, common median reference and local whitening; matched-filtering detection with a
prototype waveform learned from the data; local SVD features; iterative HDBSCAN splits; template
cleaning and merging; circus-omp template matching over the whole recording; final merges
(``auto_merge_units``). Not yet: motion correction.

Examples
--------
>>> from dsp_kitchen.synapse.ml import spykingcircus2
>>> config = spykingcircus2.Config()
>>> result = spykingcircus2.run(recording, probe, config)
>>> sorting = result.to_sorting_output(probe)
"""

from __future__ import annotations

from typing import Optional

from dsp_kitchen_bindings import (
    ProbeLayout,
    Recording,
    Spykingcircus2Config as Config,
    Spykingcircus2Result,
    spykingcircus2_provenance as provenance,
)
from dsp_kitchen_bindings import run_spykingcircus2 as _run

from ...progress import progress_callback
from .kilosort4 import Progress


def run(
    recording: Recording,
    probe: ProbeLayout,
    config: Config,
    *,
    progress: Progress = True,
    runtime: Optional[str] = None,
) -> Spykingcircus2Result:
    """Sorts a whole recording with SpyKING CIRCUS 2.

    Parameters
    ----------
    recording : Recording
        The recording to sort (channels in the probe's order).
    probe : ProbeLayout
        Contact positions (µm) of the recording's channels.
    config : Config
        Settings (see ``Config``: every setting, with SpyKING CIRCUS 2's defaults).
    progress : bool or callable, default True
        ``True``: a progress bar per stage; ``False``: none; a callable receives
        ``(stage, step, steps, done, total, unit)``.
    runtime : str, optional
        Compute runtime; default: the current one (``dsp_kitchen.runtime``).

    Returns
    -------
    Spykingcircus2Result
        Spikes (sample, unit, scaling), unit templates, noise levels, the preprocessing.
    """
    callback = progress_callback(progress)
    try:
        return _run(recording, probe, config, progress=callback, runtime=runtime)
    finally:
        if hasattr(callback, "close"):
            callback.close()


__all__ = ["Config", "Spykingcircus2Result", "provenance", "run"]
