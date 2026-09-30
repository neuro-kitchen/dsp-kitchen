# %% [markdown]
# # dsp-kitchen: NWB Zarr v3 (`15-25-33_meps.nwb.zarr`) Reader & Visualization
# Interactive notebook workflow:
# 1. Discover continuous `ElectricalSeries` and `TimeSeries` in `/acquisition`
# 2. Open the 32-channel Diaphragm HD-EMG series (`HDEMG`) via `dsp-io` (`NwbZarrRecording`)
# 3. Read a time window from the Zarr v3 store (automatically scaled to μV)
# 4. Plot single-channel and multi-channel stacked traces of the dataset

# %% [1] Imports and Workspace Resolution
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import dsp_kitchen
import dsp_kitchen.synapse as syn
from dsp_kitchen.io import NwbZarrRecording, list_nwb_series
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter, NotchFilter
from dsp_kitchen.spatial import CommonAverageReference

local_path = dsp_kitchen.get_local_path()
nwb_path = local_path / "playground" / "data" / "nwb" / "15-25-33_meps.nwb.zarr"

print(f"Workspace Base:      {local_path}")
print(f"dsp-kitchen Version: {dsp_kitchen.__version__}")
print(f"NWB Zarr Store:      {nwb_path}")

# %% [2] Inspect Available Continuous Series in NWB Store
series_list = list_nwb_series(str(nwb_path))
print(f"Found {len(series_list)} continuous series in {nwb_path.name}:")
for s in series_list:
    print(
        f"  {s['path']:20s} | {s['neurodata_type']:16s} | "
        f"channels={s['channels']:2d} | samples={s['samples']:,}"
    )

# %% [3] Open Primary ElectricalSeries (/acquisition/HDEMG) & Read Portion
rec = dsp_kitchen.open_nwb_zarr(nwb_path, series="/acquisition/HDEMG")

print(f"\nRecording Summary:")
print(f"  Name:          {rec.name}")
print(f"  Series Path:   {rec.series}")
print(f"  Shape:         {rec.shape} (channels, samples)")
print(f"  Sample Rate:   {rec.sample_rate:.4f} Hz")
print(f"  Duration:      {rec.duration_sec:.2f} s")
print(f"  Unit:          {rec.unit}")
print(f"  Channels:      {rec.channel_names[:4]} ... {rec.channel_names[-1]}")

# Read a 200 ms window starting at t = 1.0 s across all 32 HD-EMG channels
start_sec = 1.0
duration_sec = 0.200
chunk = rec.read_window(start_sec=start_sec, duration_sec=duration_sec)
num_channels, num_samples = chunk.shape
time_ms = np.arange(num_samples) / rec.sample_rate * 1000.0

print(f"\nLoaded Window ({start_sec:.2f}s .. {start_sec + duration_sec:.2f}s):")
print(f"  Chunk Shape:   {chunk.shape} (dtype={chunk.dtype})")
print(f"  Ch 0 Min/Max:  {chunk[0].min():.2f} / {chunk[0].max():.2f} {rec.unit}")
print(f"  Ch 0 RMS:      {float(np.sqrt(np.mean(chunk[0] ** 2))):.2f} {rec.unit}")

# %% [4] Plot a Portion of the NWB Zarr Dataset
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Top panel: Single-channel trace (Channel 0)
inspect_ch = 0
ch_label = rec.channel_names[inspect_ch] if rec.channel_names else f"Ch {inspect_ch}"
axes[0].plot(
    time_ms,
    chunk[inspect_ch],
    color="#1f77b4",
    linewidth=1.1,
    label=f"{ch_label} (Raw NWB Zarr)",
)
axes[0].set_title(
    f"{rec.name} — Single Channel Trace ({ch_label}, t = {start_sec:.2f}s..{start_sec + duration_sec:.2f}s)",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_ylabel(f"Amplitude ({rec.unit})", fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].legend(loc="upper right")

# Bottom panel: Stacked multi-channel waterfall (first 8 channels)
n_plot_ch = min(8, num_channels)
subset = chunk[:n_plot_ch]
std_est = float(np.std(subset))
offset_step = std_est * 4.0 if std_est > 0 else 100.0

for ch_idx in range(n_plot_ch):
    name = (
        rec.channel_names[ch_idx] if ch_idx < len(rec.channel_names) else f"Ch {ch_idx}"
    )
    axes[1].plot(
        time_ms,
        subset[ch_idx] + ch_idx * offset_step,
        linewidth=0.9,
        label=name,
    )

axes[1].set_title(
    f"Multi-Channel Stacked View (First {n_plot_ch} of {num_channels} HD-EMG Channels)",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlabel(f"Time from {start_sec:.2f} s (ms)", fontsize=11)
axes[1].set_ylabel(f"Stacked Channels ({rec.unit})", fontsize=10)
axes[1].set_yticks([i * offset_step for i in range(n_plot_ch)])
axes[1].set_yticklabels(
    [
        rec.channel_names[i] if i < len(rec.channel_names) else f"Ch {i}"
        for i in range(n_plot_ch)
    ]
)
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

# %% [5] In-VRAM GPU Filtering Pipeline (Bandpass + 60 Hz Notch + CAR)
pipeline = Pipeline(
    [
        BandpassFilter(
            low_hz=300.0, high_hz=3000.0
        ),  # 4th-order Butterworth HD-EMG / MEP band
        NotchFilter(freq_hz=60.0, q=30.0),  # 60 Hz power-line hum rejection
        CommonAverageReference(),  # Spatial CAR across the 32-ch grid
    ]
)

print("Constructed In-VRAM Pipeline:")
for idx, stage in enumerate(pipeline.stages):
    print(f"  Stage {idx + 1}: {stage}")

filtered_chunk = pipeline.run(
    np.ascontiguousarray(chunk, dtype=np.float32), fs=rec.sample_rate
)

print(f"\n[Filtered HD-EMG Output]")
print(f"  Input Shape:    {chunk.shape}")
print(f"  Output Shape:   {filtered_chunk.shape}")
print(
    f"  Raw Ch0 RMS:    {float(np.sqrt(np.mean(chunk[inspect_ch] ** 2))):.2f} {rec.unit}"
)
print(
    f"  Filt Ch0 RMS:   {float(np.sqrt(np.mean(filtered_chunk[inspect_ch] ** 2))):.2f} {rec.unit}"
)

# %% [6] Plot Raw vs. Filtered Comparison (Single-Channel & Multi-Channel Stacked)
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Top panel: Raw vs. Filtered overlay on Channel 0
axes[0].plot(
    time_ms,
    chunk[inspect_ch],
    color="#1f77b4",
    alpha=0.55,
    linewidth=1.0,
    label=f"{ch_label} Raw",
)
axes[0].plot(
    time_ms,
    filtered_chunk[inspect_ch],
    color="#2ca02c",
    linewidth=1.2,
    label=f"{ch_label} Filtered (Bandpass 300-3000 Hz + 60 Hz Notch + CAR)",
)
axes[0].set_title(
    f"{rec.name} — Raw vs. Filtered Comparison ({ch_label})",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_ylabel(f"Amplitude ({rec.unit})", fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].legend(loc="upper right")

# Bottom panel: Filtered multi-channel stacked waterfall (first 8 channels)
filt_subset = filtered_chunk[:n_plot_ch]
filt_std = float(np.std(filt_subset))
filt_offset_step = filt_std * 4.0 if filt_std > 0 else 100.0

for ch_idx in range(n_plot_ch):
    name = (
        rec.channel_names[ch_idx] if ch_idx < len(rec.channel_names) else f"Ch {ch_idx}"
    )
    axes[1].plot(
        time_ms,
        filt_subset[ch_idx] + ch_idx * filt_offset_step,
        linewidth=0.9,
        label=name,
    )

axes[1].set_title(
    f"Filtered Multi-Channel Stacked View (First {n_plot_ch} Channels)",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlabel(f"Time from {start_sec:.2f} s (ms)", fontsize=11)
axes[1].set_ylabel(f"Stacked Filtered ({rec.unit})", fontsize=10)
axes[1].set_yticks([i * filt_offset_step for i in range(n_plot_ch)])
axes[1].set_yticklabels(
    [
        rec.channel_names[i] if i < len(rec.channel_names) else f"Ch {i}"
        for i in range(n_plot_ch)
    ]
)
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

# %% [7] Out-of-Core Threshold Spike Sorting (Lazy Slicing + CubeCL GPU Streaming Engine)
# Define 32-channel 4x8 HD-EMG planar grid layout (100 um pitch)
grid_rows, grid_cols = 4, 8
pitch_um = 100.0
grid_positions = [
    (float(c) * pitch_um, float(r) * pitch_um)
    for r in range(grid_rows)
    for c in range(grid_cols)
]
probe = syn.custom_layout("HDEMG-32-Grid", grid_positions[: rec.channels])

threshold_factor = 5.0
pre_ms = 1.0
post_ms = 2.0

# Lazy zero-load slicing: slice any time window or sample range without loading into RAM!
# - Use `rec.slice_time(start_sec=0.0, duration_sec=60.0)` or `rec[:, 0:1_000_000]` for a subset
# - Or pass `rec` directly to stream the entire 2,899.9 s recording
target_rec = rec.slice_time(start_sec=0.0, duration_sec=3000)

print(
    f"Pipeline filter settling at {target_rec.sample_rate:.1f} Hz: "
    f"{pipeline.settling_samples(target_rec.sample_rate)} samples"
)
print(
    f"Streaming {target_rec} ({target_rec.samples:,} samples, {target_rec.duration_sec:.1f} s) in Rust + CubeCL WGPU..."
)

sort_res = syn.sort_recording(
    target_rec,
    pipeline=pipeline,
    probe=probe,
    threshold_factor=threshold_factor,
    refractory_ms=1.0,
    spatial_radius_um=150.0,
    k_neighbors=4,
    pre_ms=pre_ms,
    post_ms=post_ms,
    batch_duration_sec=20.0,
    calibration_duration_sec=5.0,
    apply_sinc_shift=True,
)

templates = sort_res.all_templates()
spike_counts = sort_res.channel_spike_counts
active_channels = sum(1 for t in templates if t is not None)

print(f"\n[Out-of-Core Streaming Sort Summary]")
print(f"  Dynamic Halos (L, R):  {sort_res.halos} samples")
print(f"  Calibrated Mean σ_n:   {np.mean(sort_res.channel_sigmas_uv):.2f} {rec.unit}")
print(f"  Total Raw Crossings:   {sort_res.total_raw_crossings:,}")
print(f"  Total Deduped Spikes:  {sort_res.total_dedup_spikes:,}")
print(f"  Active Grid Channels:  {active_channels} / {rec.channels}")

# %% [8] Plot Extracted Templates in a 4x8 Electrode Grid
fig, axes = plt.subplots(
    grid_rows,
    grid_cols,
    figsize=(16, 8.5),
    sharex=True,
    sharey=True,
)

first_valid = next((t for t in templates if t is not None), None)
snippet_len = (
    first_valid["num_samples"]
    if first_valid is not None
    else int((pre_ms + post_ms) * 1e-3 * rec.sample_rate)
)
pre_samples = int(round(pre_ms * 1e-3 * rec.sample_rate))
snip_time_ms = (np.arange(snippet_len) - pre_samples) / rec.sample_rate * 1000.0

for ch in range(rec.channels):
    r, c = divmod(ch, grid_cols)
    ax = axes[r, c]
    ch_name = rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch {ch}"
    n_spk = spike_counts[ch]
    tmpl = templates[ch]

    if tmpl is not None:
        # Row 0 of [k_neighbors, num_samples] is the primary channel waveform
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

    if r == grid_rows - 1:
        ax.set_xlabel("Time (ms)", fontsize=8)
    if c == 0:
        ax.set_ylabel(f"Amplitude ({rec.unit})", fontsize=8)

fig.suptitle(
    f"{rec.name} — Out-of-Core Spike Templates Across 4×8 HD-EMG Grid "
    f"(-{threshold_factor:.1f}σ, Total Spikes={sort_res.total_dedup_spikes:,})",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

# %%
