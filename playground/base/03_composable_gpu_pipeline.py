# %% [markdown]
# # dsp-kitchen (`dsp-base` & `dsp-io`): Composable In-VRAM Pipelines, Probe Layouts & NWB Zarr I/O
# Consolidates the multi-channel pipeline, bridge verification, and NWB Zarr reader workflows:
# 1. **Standard & Custom Probe Geometries (`ProbeLayout`)**:
#    - `neuropixels_1_0_layout`, `neuropixels_2_0_layout`, `tetrode_layout`, `utah_array_layout`, `custom_layout`
# 2. **NWB Zarr v3 Discovery & Windowed Reading (`NwbZarrRecording`)**:
#    - `list_nwb_series`, `open_nwb_zarr`, `read_window`, lazy `slice_time`
# 3. **Composable In-VRAM GPU Pipeline (`Pipeline` & `DspSession`)**:
#    - Multi-stage execution (`Scale` $\to$ `BandpassFilter` $\to$ `NotchFilter` $\to$ `CommonAverageReference`)
#    - 32-channel vs. 384-channel hardware scaling evaluation

# %% [1] Imports & Workspace Setup
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.io import list_nwb_series
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter, NotchFilter
from dsp_kitchen.filter.non_linear import MedianFilter
from dsp_kitchen.spatial import CommonAverageReference
from dsp_kitchen.math import Clamp, Scale

local_path = dk.get_local_path()
nwb_path = local_path / "playground" / "data" / "nwb" / "15-25-33_meps.nwb.zarr"

print(f"Workspace Base:      {local_path}")
print(f"dsp-kitchen Version: {dk.__version__}")

# %% [2] Probe Layouts: Neuropixels 1.0/2.0, Tetrode, Utah Array & Custom Grids
np1 = dk.neuropixels_1_0_layout()
np2 = dk.neuropixels_2_0_layout()
tet = dk.tetrode_layout()
utah = dk.utah_array_layout()

print("[Built-in Probe Geometries]")
for p in (np1, np2, tet, utah):
    pos = np.asarray(p.contact_positions())
    print(
        f"  {p.name:20s} | channels={p.total_channels:3d} | "
        f"x=[{pos[:, 0].min():.0f}, {pos[:, 0].max():.0f}] um, "
        f"y=[{pos[:, 1].min():.0f}, {pos[:, 1].max():.0f}] um"
    )

# %% [3] Discover & Read NWB Zarr v3 Series (`15-25-33_meps.nwb.zarr`)
if nwb_path.exists():
    series_list = list_nwb_series(str(nwb_path))
    print(f"\nFound {len(series_list)} continuous series in {nwb_path.name}:")
    for s in series_list:
        print(
            f"  {s['path']:20s} | {s['neurodata_type']:16s} | "
            f"channels={s['channels']:2d} | samples={s['samples']:,}"
        )

    rec = dk.open_nwb_zarr(nwb_path, series="/acquisition/HDEMG")
    print(
        f"\nOpened Series: {rec.name} ({rec.shape}, {rec.sample_rate:.2f} Hz, unit={rec.unit})"
    )

    # Read a 200 ms window starting at t = 1.0 s across all 32 HD-EMG channels
    chunk_32 = rec.read_window(start_sec=1.0, duration_sec=0.200)
    fs = rec.sample_rate
    unit = rec.unit
    ch_names = rec.channel_names
else:
    print(f"\n[Note] {nwb_path} not found; generating synthetic 32-channel chunk.")
    fs = 24_414.0625
    unit = "uV"
    chunk_32 = np.random.randn(32, int(0.200 * fs)).astype(np.float32) * 35.0
    ch_names = [f"Ch {i}" for i in range(32)]

# %% [4] Execute Composable In-VRAM Pipeline on 32-Channel Recording
pipeline = Pipeline(
    [
        BandpassFilter(
            low_hz=300.0, high_hz=3000.0, order=4, direction="forward-backward"
        ),
        NotchFilter(freq_hz=60.0, q=30.0, direction="forward-backward"),
        MedianFilter(),
        CommonAverageReference(),
        Clamp(min_val=-2000.0, max_val=2000.0),
    ]
)

print("\nConfigured In-VRAM Pipeline:")
for idx, stage in enumerate(pipeline.stages):
    print(f"  Stage {idx + 1}: {stage}")
print(
    f"  Required boundary settling halos at {fs:.1f} Hz: {pipeline.settling(fs)} samples"
)

filtered_32 = pipeline.run(np.ascontiguousarray(chunk_32, dtype=np.float32), fs=fs)

print(f"\n[32-Channel Execution]")
print(f"  Input Shape:   {chunk_32.shape} -> Output Shape: {filtered_32.shape}")
print(f"  Raw Ch 0 RMS:  {float(np.sqrt(np.mean(chunk_32[0] ** 2))):.2f} {unit}")
print(f"  Filt Ch 0 RMS: {float(np.sqrt(np.mean(filtered_32[0] ** 2))):.2f} {unit}")

# %% [5] 384-Channel High-Density Neuropixels Scaling & DspSession Check
# Tile the 32-channel recording to 384 channels (12x) to evaluate full Neuropixels 1.0 probe throughput
chunk_384 = np.tile(chunk_32[:, :1500], (12, 1)).astype(np.float32)
filtered_384 = pipeline.run(chunk_384, fs=30_000.0)

session = dk.DspSession(sample_rate=30_000.0, channels=384)
session_out = session.run_pipeline_wgpu(chunk_384)

print(f"\n[384-Channel Neuropixels 1.0 Scaling]")
print(f"  Session Info:        {session.info()}")
print(f"  384-ch Pipeline Out: {filtered_384.shape}, dtype={filtered_384.dtype}")
print(f"  DspSession Out:      {session_out.shape}, dtype={session_out.dtype}")

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
    axes[0].set_ylabel(f"Amplitude ({unit})")
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
    axes[1].set_ylabel(f"Stacked ({unit})")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

# %%
