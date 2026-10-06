# %% [markdown]
# # dsp-kitchen (`dsp-synapse-ml`): EMUsort Front End on HD-EMG
# EMUsort (O'Connell et al., 2026), a Kilosort4 fork for motor units. `run_front_end` runs the
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
from dsp_kitchen.synapse.ml import emusort, kilosort4

# Test data lives outside the playground: `<repository>/data/`, or the folder in DSP_KITCHEN_DATA
DATA_DIR = Path(os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / "data"))
NWB_PATH = DATA_DIR / "nwb" / "15-25-33_meps.nwb.zarr"
GRID_ROWS, GRID_COLS = 4, 8
ELECTRODE_PITCH_UM = 100.0  # inter-electrode distance of the grid (set your array's)

print(emusort.provenance().citation())
hdemg = next((s["id"] for s in list_sources(str(NWB_PATH)) if "HDEMG" in s["id"]), None)
rec = Recording(str(NWB_PATH), source=hdemg)
probe = syn.ProbeLayout.hdemg_grid("HD-EMG 4x8", GRID_ROWS, GRID_COLS, ELECTRODE_PITCH_UM)
config = emusort.Config()
print(f"\n{rec}\n{config}")

# %% [2] Front End over the Whole Recording
result = kilosort4.run_front_end(rec, probe, config)
spikes = result.spikes()
delays, reference = result.channel_delays
print(f"\n{result} on {dk.runtime.current()}")
print(f"Channel delays (samples) vs channel {reference}: {delays}")
print(f"{len(spikes['sample']):,} spikes; amplitude (whitened) median {np.median(spikes['amplitude']):.1f}")

# %% [3] Plot
if HAS_PLT:
    ks4 = config.kilosort4
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    t_ms = (np.arange(ks4.nt) - ks4.resolved_nt0min()) / rec.sample_rate * 1e3
    for row in result.templates.wtemp:
        axes[0].plot(t_ms, row, linewidth=1.2)
    axes[0].set_title("Universal templates (learned)", fontweight="bold")
    axes[0].set_xlabel("Time from peak (ms)")
    axes[1].imshow(np.reshape(delays, (GRID_ROWS, GRID_COLS)) / rec.sample_rate * 1e3, cmap="coolwarm")
    axes[1].set_title("Channel delays (ms)", fontweight="bold")
    axes[2].scatter(np.asarray(spikes["sample"]) / rec.sample_rate, spikes["y_um"], s=3, c=spikes["template"], cmap="tab10")
    axes[2].set_title("Spikes: time vs position", fontweight="bold")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("y (µm)")
    plt.tight_layout()
    plt.show()

# %%
