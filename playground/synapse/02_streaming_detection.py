# %% [markdown]
# # dsp-kitchen (`dsp-synapse`): Streaming Detection of a Whole Recording
# `syn.detect_recording` processes a recording of any length in Rust, on the device, with bounded
# memory (no clustering: one template per primary channel):
# 1. Open the HD-EMG signal of an NWB Zarr store (`io.Recording`); lazy `slice_time` windows
# 2. Halos sized from the pipeline's settling and the snippet window, so any batch size gives
#    the whole-recording result
# 3. Noise calibration, detection, deduplication, realigned snippets and per-channel templates
# 4. Templates plotted across the 4 × 8 grid
#
# Data (local, git-ignored): `data/nwb/15-25-33_meps.nwb.zarr` (see playground/README.md).

# %% [1] Imports & Open the Recording
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
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter, NotchFilter
from dsp_kitchen.spatial import CommonAverageReference

# Test data lives outside the playground: `<repository>/data/`, or the folder in DSP_KITCHEN_DATA
DATA_DIR = Path(os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / "data"))
NWB_PATH = DATA_DIR / "nwb" / "15-25-33_meps.nwb.zarr"
GRID_ROWS, GRID_COLS = 4, 8
ELECTRODE_PITCH_UM = 100.0  # inter-electrode distance of the grid (set your array's)
WINDOW = (0.0, 60.0)  # (start, duration) in seconds; None for the whole recording
LINE_HZ, NOTCH_Q = 60.0, 30.0

hdemg = next((s["id"] for s in list_sources(str(NWB_PATH)) if "HDEMG" in s["id"]), None)
rec = Recording(str(NWB_PATH), source=hdemg)
unit = rec.units[0]
print(f"Opened {rec}")

# %% [2] Pre-processing Pipeline & Probe
pipeline = Pipeline([BandpassFilter(300.0, 3000.0, order=4), NotchFilter(LINE_HZ, NOTCH_Q), CommonAverageReference()])
probe = syn.ProbeLayout.hdemg_grid("HD-EMG 4x8", GRID_ROWS, GRID_COLS, ELECTRODE_PITCH_UM)
print(f"Pipeline settling at {rec.sample_rate:.2f} Hz: {pipeline.settling(fs=rec.sample_rate)} samples")

# %% [3] Streaming Detection (Rust, on the device, bounded memory)
# `slice_time` makes a lazy view; nothing is read until detection streams it.
target_rec = rec.slice_time(*WINDOW) if WINDOW else rec
print(f"Detecting in {target_rec.samples:,} samples ({target_rec.duration_sec:.1f} s) on {dk.runtime.current()}...")

result = syn.detect_recording(target_rec, pipeline, probe, spatial_radius_um=1.5 * ELECTRODE_PITCH_UM)
templates = [result.template(ch) for ch in range(result.channels)]
spike_counts = result.channel_spike_counts
active_channels = sum(1 for t in templates if t is not None)

print("\n[Streaming detection]")
print(f"  {result}")
print(f"  Halos (left, right): {result.halos} samples")
print(f"  Mean noise σ:        {np.mean(result.noise_sigmas):.3g} {unit}")
print(f"  Crossings / spikes:  {result.total_crossings:,} / {result.total_spikes:,}")
print(f"  Active channels:     {active_channels} / {result.channels}")

# %% [4] Plot Welford-Accumulated Templates Across the 4×8 Electrode Grid
if HAS_PLT:
    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLS,
        figsize=(16, 8.5),
        sharex=True,
        sharey=True,
    )

    first_valid = next((t for t in templates if t is not None), None)
    # The streaming defaults cut snippets 1 ms before and 2 ms after the trough
    pre_samples = int(round(1e-3 * rec.sample_rate))
    snippet_len = first_valid["mean"].shape[1] if first_valid is not None else 3 * pre_samples
    snip_time_ms = (np.arange(snippet_len) - pre_samples) / rec.sample_rate * 1000.0

    for ch in range(result.channels):
        r, c = divmod(ch, GRID_COLS)
        ax = axes[r, c]
        ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch {ch}"
        n_spk = spike_counts[ch]
        tmpl = templates[ch]

        if tmpl is not None:
            mean_wf = tmpl["mean"][0]
            std_wf = tmpl["std"][0]
            ax.fill_between(
                snip_time_ms,
                mean_wf - std_wf,
                mean_wf + std_wf,
                color="#2ca02c",
                alpha=0.25,
            )
            ax.plot(snip_time_ms, mean_wf, color="#1b7837", linewidth=1.5)
        else:
            ax.plot(
                snip_time_ms,
                np.zeros_like(snip_time_ms),
                color="#999999",
                linestyle="--",
                linewidth=0.8,
            )

        ax.axvline(0.0, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{ch_name} (n={n_spk:,})", fontsize=9, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

        if r == GRID_ROWS - 1:
            ax.set_xlabel("Time (ms)", fontsize=8)
        if c == 0:
            ax.set_ylabel(f"Amplitude ({unit or 'a.u.'})", fontsize=8)

    fig.suptitle(
        f"{rec.name} — streaming detection templates across the 4×8 grid "
        f"(spikes={result.total_spikes:,})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
