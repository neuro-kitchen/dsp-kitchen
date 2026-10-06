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
    run_emusort as run,
)

__all__ = [
    "Config",
    "apply_channel_delays",
    "estimate_channel_delays",
    "provenance",
    "run",
]
