# %% [markdown]
# # dsp-kitchen (`dsp-base` & `dsp-io`): Device Pipelines, Probe Layouts & Recordings
# 1. **Probe geometries (`ProbeLayout`)**: Neuropixels 1.0 / 2.0, tetrode, Utah array presets.
# 2. **Recordings (`dsp_kitchen.io`)**: `list_sources` (the signals a file holds) and `Recording`
#    (any format dsp-io reads; lazy slices, reads in each channel's unit).
# 3. **Device pipeline (`Pipeline`)**: band-pass → notch → median → common reference → clamp, run
#    on the device without intermediate copies; 32 and 384 channels.
#
# Data (local, git-ignored): `data/nwb/15-25-33_meps.nwb.zarr` (see playground/README.md); a synthetic chunk is
# used when it is absent.

# %% [1] Imports & Paths
import os
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
from dsp_kitchen.io import Recording, list_sources
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter, NotchFilter
from dsp_kitchen.filter.non_linear import MedianFilter
from dsp_kitchen.spatial import CommonAverageReference
from dsp_kitchen.math import Clamp
from dsp_kitchen.synapse import ProbeLayout

# Test data lives outside the playground: `<repository>/data/`, or the folder in DSP_KITCHEN_DATA
DATA_DIR = Path(os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / "data"))
NWB_PATH = DATA_DIR / "nwb" / "15-25-33_meps.nwb.zarr"
WINDOW = (1.0, 0.200)  # (start, duration) in seconds
LINE_HZ, NOTCH_Q = 60.0, 30.0

print(f"dsp-kitchen {dk.__version__}, runtimes {dk.runtime.available()} (using {dk.runtime.current()})")

# %% [2] Probe Layouts
print("[Probe presets]")
for p in (ProbeLayout.neuropixels_1_0(), ProbeLayout.neuropixels_2_0(), ProbeLayout.tetrode(), ProbeLayout.utah_array()):
    pos = np.asarray(p.contact_positions())
    print(
        f"  {p.name:28s} | channels={p.total_channels:3d} | "
        f"x=[{pos[:, 0].min():.0f}, {pos[:, 0].max():.0f}] µm, y=[{pos[:, 1].min():.0f}, {pos[:, 1].max():.0f}] µm"
    )

# %% [3] Discover & Read a Recording
if NWB_PATH.exists():
    sources = list_sources(str(NWB_PATH))
    print(f"\n{len(sources)} signal(s) in {NWB_PATH.name}:")
    for s in sources:
        print(f"  {s['id']:24s} | {s['kind']:10s} | {s['channels']:2d} ch × {s['samples']:,} @ {s['sample_rate']:.2f} Hz ({s['unit']})")
    hdemg = next((s["id"] for s in sources if "HDEMG" in s["id"]), None)
    rec = Recording(str(NWB_PATH), source=hdemg)
    print(f"\nOpened {rec}")
    chunk_32 = rec.read_window(*WINDOW)
    fs, unit, ch_names = rec.sample_rate, rec.units[0], rec.channel_names
else:
    print(f"\n[Note] {NWB_PATH} not found; using a synthetic 32-channel chunk (arbitrary units).")
    fs, unit = 24_414.0625, ""
    chunk_32 = np.random.default_rng(0).standard_normal((32, int(WINDOW[1] * fs))).astype(np.float32)
    ch_names = [f"ch{i}" for i in range(32)]

# %% [4] Device Pipeline on 32 Channels
clip = float(np.abs(chunk_32).max())
pipeline = Pipeline(
    [
        BandpassFilter(300.0, 3000.0, order=4),
        NotchFilter(LINE_HZ, NOTCH_Q),
        MedianFilter(),
        CommonAverageReference(),
        Clamp(-clip, clip),
    ]
)
print("\nPipeline stages:")
for idx, stage in enumerate(pipeline.stages):
    print(f"  {idx + 1}: {stage}")
print(f"  Settling halos at {fs:.1f} Hz: {pipeline.settling(fs=fs)} samples")

filtered_32 = pipeline.run(chunk_32, fs=fs)
print(f"\n[32 channels] {chunk_32.shape} -> {filtered_32.shape}")
print(f"  ch0 RMS raw {float(np.sqrt(np.mean(chunk_32[0] ** 2))):.2f} {unit}, filtered {float(np.sqrt(np.mean(filtered_32[0] ** 2))):.2f} {unit}")

# %% [5] 384 Channels (Neuropixels 1.0 width)
chunk_384 = np.tile(chunk_32, (12, 1)).astype(np.float32)
filtered_384 = pipeline.run(chunk_384, fs=fs)
print(f"\n[384 channels] {chunk_384.shape} -> {filtered_384.shape}, dtype={filtered_384.dtype}")

# %% [6] Plot Raw vs. Filtered Multi-Channel Waterfall
if HAS_PLT:
    time_ms = np.arange(chunk_32.shape[1]) / fs * 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Top panel: Single channel overlay
    axes[0].plot(
        time_ms,
        chunk_32[0],
        color="#1f77b4",
        alpha=0.55,
        linewidth=1.0,
        label=f"{ch_names[0]} Raw",
    )
    axes[0].plot(
        time_ms,
        filtered_32[0],
        color="#2ca02c",
        linewidth=1.2,
        label=f"{ch_names[0]} Filtered",
    )
    axes[0].set_title(
        f"Single-Channel Trace Comparison ({ch_names[0]})", fontweight="bold"
    )
    axes[0].set_ylabel(f"Amplitude ({unit or 'a.u.'})")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Bottom panel: First 8 channels stacked waterfall
    n_plot = min(8, chunk_32.shape[0])
    step = float(np.std(filtered_32[:n_plot])) * 4.0 or 100.0
    for ch in range(n_plot):
        axes[1].plot(
            time_ms, filtered_32[ch] + ch * step, linewidth=0.9, label=ch_names[ch]
        )
    axes[1].set_yticks([i * step for i in range(n_plot)])
    axes[1].set_yticklabels(ch_names[:n_plot])
    axes[1].set_title(
        f"Filtered Multi-Channel Stacked View (First {n_plot} Channels)",
        fontweight="bold",
    )
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel(f"Stacked ({unit or 'a.u.'})")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

# %%
