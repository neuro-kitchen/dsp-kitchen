"""
EMUsort (O'Connell et al., 2026), a Kilosort4 fork for motor units, written from the paper:
channel-delay removal, several clip thresholds and HDBSCAN outlier removal before the universal
templates. ``emusort.run(recording, probe, emusort.Config())`` runs EMUsort's pipeline
(no common reference, delay removal, universal templates, spike detection) over a whole recording.
"""

from dsp_kitchen_bindings import (
    ChannelDelayEstimator,
    EmusortConfig as Config,
    EmusortResult,
    apply_channel_delays,
    emusort_provenance as provenance,
    run_emusort as run,
)

__all__ = [
    "ChannelDelayEstimator",
    "Config",
    "EmusortResult",
    "apply_channel_delays",
    "provenance",
    "run",
]
