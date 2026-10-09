"""EMUsort (O'Connell et al., *eLife* 2026), a Kilosort4 fork for motor units, written from the paper,
running on the GPU.

EMUsort runs Kilosort4's stages (see ``kilosort4``) with its own settings and two additions:
channel-delay removal (a MUAP reaches channels at different times) and HDBSCAN outlier removal
before the universal templates are learned; it pools clips over several thresholds and skips the
common average reference. ``estimate_channel_delays`` and ``apply_channel_delays`` expose the delay
stage on its own. Not yet: the refractory criteria, the final merges, EMUsort's unit score.

Examples
--------
>>> from dsp_kitchen.synapse.ml import emusort
>>> config = emusort.Config()
>>> config.nt = 121                        # 5 ms at 24.4 kHz: match the MUAP width
>>> result = emusort.run(recording, probe, config)
>>> delays, reference = result.channel_delays
"""

from __future__ import annotations

from typing import Optional

from dsp_kitchen_bindings import (
    EmusortConfig as Config,
    Kilosort4Result,
    ProbeLayout,
    Recording,
    apply_channel_delays,
    estimate_channel_delays,
    emusort_provenance as provenance,
)
from dsp_kitchen_bindings import run_emusort as _run

from ...progress import progress_callback
from .kilosort4 import Progress


def run(
    recording: Recording,
    probe: ProbeLayout,
    config: Config,
    *,
    preprocessing_from: Optional[Kilosort4Result] = None,
    progress: Progress = True,
    runtime: Optional[str] = None,
) -> Kilosort4Result:
    """Sorts a whole recording with EMUsort.

    Parameters
    ----------
    recording : Recording
        The recording to sort (e.g. an HD-EMG or Myomatrix array; channels in the probe's order).
    probe : ProbeLayout
        Contact positions (µm) of the recording's channels.
    config : Config
        Settings (see ``Config``: every setting, with EMUsort's defaults). Check ``config.nt``
        against the MUAP width of the data: it counts samples.
    preprocessing_from : Kilosort4Result, optional
        An earlier result on the same recording with the same fit settings: its preprocessing and
        channel delays are reused.
    progress : bool or callable, default True
        ``True``: a progress bar per stage; ``False``: none; a callable receives
        ``(stage, step, steps, done, total, unit)``.
    runtime : str, optional
        Compute runtime; default: the current one (``dsp_kitchen.runtime``).

    Returns
    -------
    Kilosort4Result
        As Kilosort4's, with ``channel_delays`` set; spike times are in the reference channel's
        frame.
    """
    callback = progress_callback(progress)
    try:
        return _run(recording, probe, config, preprocessing_from=preprocessing_from, progress=callback, runtime=runtime)
    finally:
        if hasattr(callback, "close"):
            callback.close()


__all__ = [
    "Config",
    "apply_channel_delays",
    "estimate_channel_delays",
    "provenance",
    "run",
]
