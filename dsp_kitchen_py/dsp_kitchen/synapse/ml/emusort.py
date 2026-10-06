"""
EMUsort (O'Connell et al., 2026), a Kilosort4 fork for motor units, written from the paper:
channel-delay removal, several clip thresholds and HDBSCAN outlier removal before the universal
templates. ``emusort.run(recording, probe, emusort.Config())`` runs Kilosort4's runner with
EMUsort's settings (no common reference, delay removal, universal templates, spike detection) over
a whole recording and returns a ``kilosort4.Kilosort4Result``. ``estimate_channel_delays`` (all
batches in one call, read back once) and ``apply_channel_delays`` run on the device.
"""

from dsp_kitchen_bindings import (
    EmusortConfig as Config,
    apply_channel_delays,
    estimate_channel_delays,
    emusort_provenance as provenance,
)
from dsp_kitchen_bindings import run_emusort as _run

from ...progress import progress_callback


def run(recording, probe, config, *, preprocessing_from=None, progress=True, runtime=None):
    """EMUsort over the whole ``recording`` (Kilosort4's runner with EMUsort's settings, channel
    delays and outlier removal), showing a progress bar per stage with the time left
    (``progress=False``: none; a callable receives ``(stage, step, steps, done, total, unit)``)."""
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
