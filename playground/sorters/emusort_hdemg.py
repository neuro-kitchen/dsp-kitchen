# %% [markdown]
# # dsp-kitchen (`dsp-synapse-ml`): EMUsort on HD-EMG
# EMUsort (O'Connell et al., 2026), a Kilosort4 fork for motor units. `emusort.run` runs the
# stages implemented so far over the whole recording, in Rust and on the device, in halo windows
# of `batch_size` (bounded memory):
# 1. High-pass, local whitening (EMUsort: no common reference)
# 2. Channel-delay removal (a MUAP reaches channels at different times; ±2 ms)
# 3. Universal templates learned at 6–15 σ, HDBSCAN outlier removal, 9 PCs / 9 templates
# 4. Universal-template detection
#
# Clustering, deconvolution and merging are not implemented yet.
# Data (local, git-ignored): `data/nwb/15-25-33_meps.nwb.zarr` (see playground/README.md).

# %% [1] Recording, Probe & Settings
import os
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.io import Recording, list_sources
from dsp_kitchen.synapse.ml import emusort

# Test data lives outside the playground: `<repository>/data/`, or the folder in DSP_KITCHEN_DATA
DATA_DIR = Path(
    os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / "data")
)
NWB_PATH = DATA_DIR / "nwb" / "15-25-33_meps.nwb.zarr"
GRID_ROWS, GRID_COLS = 4, 8
ELECTRODE_PITCH_UM = 100.0  # inter-electrode distance of the grid (set your array's)

print(emusort.provenance().citation())
hdemg = next((s["id"] for s in list_sources(str(NWB_PATH)) if "HDEMG" in s["id"]), None)
rec = Recording(str(NWB_PATH), source=hdemg)
rec_segment = rec.slice_time(
    start_sec=0.0, end_sec=200.0
)  # 200 s: quick to sort; set longer as needed
probe = syn.ProbeLayout.hdemg_grid(
    "HD-EMG 4x8", GRID_ROWS, GRID_COLS, ELECTRODE_PITCH_UM
)
config = emusort.Config()
# MUAPs outlast Kilosort4's 61-sample (2.5 ms at 24.4 kHz) window: measured on this recording,
# 121 samples (5 ms) removes the truncated templates and their re-matched late phases (see the
# book's EMUsort parameters page); nt0min follows (int(20 · nt / 61))
ks = config.kilosort4
ks.nt, ks.nt0min = 121, None
config.kilosort4 = ks
print(f"\n{rec}\n{config}")

# %% [2] EMUsort over the Recording
result = emusort.run(rec_segment, probe, config)
spikes = result.spikes()
delays, reference = result.channel_delays
print(f"\n{result} on {dk.runtime.current()}")
print(f"Channel delays (samples) vs channel {reference}: {delays}")
print(
    f"{len(spikes['sample']):,} spikes; amplitude (whitened) median {np.median(spikes['amplitude']):.1f}"
)

# %% [3] Export (one unit per cluster; spike times in the reference channel's frame)
sorting = result.to_sorting_output(probe)
print(
    f"\n{sorting} ({result.sorter}, {result.sample_rate_hz:.0f} Hz, {result.total_samples:,} samples)"
)
for row in sorting.summary_table()[:5]:
    print(f"  {row}")

# %% [4] Plot
if HAS_PLT:
    ks4 = config.kilosort4
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    t_ms = (np.arange(ks4.nt) - ks4.resolved_nt0min()) / rec.sample_rate * 1e3
    for row in result.templates.wtemp:
        axes[0].plot(t_ms, row, linewidth=1.2)
    axes[0].set_title("Universal templates (learned)", fontweight="bold")
    axes[0].set_xlabel("Time from peak (ms)")
    axes[1].imshow(
        np.reshape(delays, (GRID_ROWS, GRID_COLS)) / rec.sample_rate * 1e3,
        cmap="coolwarm",
    )
    axes[1].set_title("Channel delays (ms)", fontweight="bold")
    axes[2].scatter(
        np.asarray(spikes["sample"]) / rec.sample_rate,
        spikes["y_um"],
        s=3,
        c=spikes["template"],
        cmap="tab10",
    )
    axes[2].set_title("Spikes: time vs position", fontweight="bold")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("y (µm)")
    plt.tight_layout()
    plt.show()

# %% [5] Inspect One Channel
# The run's own preprocessing of a 10 s segment with the units on the chosen channel; spike times
# are in the reference channel's frame, so each channel's marks are moved by its delay.
from inspection import inspect, preprocessed_segment

INSPECT_CHANNEL, INSPECT_START, INSPECT_END = 2, 0.0, 0.2
signal, offset = preprocessed_segment(result, rec_segment, INSPECT_START, INSPECT_END)
if HAS_PLT:
    inspector = inspect(
        sorting,
        signal,
        rec_segment.sample_rate,
        offset=offset,
        channel=INSPECT_CHANNEL,
        start=INSPECT_START,
        end=INSPECT_START + 0.5,
        channel_shifts=delays,
    )

# %%
