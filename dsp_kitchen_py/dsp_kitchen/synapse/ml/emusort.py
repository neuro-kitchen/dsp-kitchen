"""
EMUsort (O'Connell et al., 2026), a Kilosort4 fork for motor units, written from the paper:
channel-delay removal, several clip thresholds and HDBSCAN outlier removal before the universal
templates. ``kilosort4.run_front_end(recording, probe, emusort.Config())`` runs EMUsort's front end
(no common reference, delay removal, its templates) over a whole recording.
"""

from dsp_kitchen_bindings import (
    ChannelDelayEstimator,
    EmusortConfig as Config,
    apply_channel_delays,
    emusort_provenance as provenance,
)

__all__ = ["ChannelDelayEstimator", "Config", "apply_channel_delays", "provenance"]
