# %% [markdown]
# # dsp-kitchen (`dsp-synapse`): Out-of-Core Streaming Spike Sorting on NWB Zarr v3
# Interactive notebook showing how to stream long recordings in constant memory using `syn.sort_recording`:
# 1. Open `/acquisition/HDEMG` from `15-25-33_meps.nwb.zarr` via `open_nwb_zarr`
# 2. Zero-copy lazy slicing with `rec.slice_time(...)` or `rec[:, start:end]`
# 3. Automatic boundary halo computation (`pipeline.settling(fs)` + template window)
# 4. CubeCL GPU streaming detection, spatial deduplication, sub-sample sinc realignment, and Welford online templates
# 5. Plot extracted multi-channel templates across the 4×8 HD-EMG electrode grid

# %% [1] Imports & Open NWB Zarr Recording
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter, NotchFilter
from dsp_kitchen.spatial import CommonAverageReference

local_path = dk.get_local_path()
nwb_path = local_path / "playground" / "data" / "nwb" / "15-25-33_meps.nwb.zarr"

rec = dk.open_nwb_zarr(nwb_path, series="/acquisition/HDEMG")
print(f"Opened NWB Zarr Recording:")
print(f"  Name:        {rec.name} ({rec.series})")
print(f"  Shape:       {rec.shape} (channels, samples)")
print(f"  Sample Rate: {rec.sample_rate:.4f} Hz")
print(f"  Duration:    {rec.duration_sec:.2f} s")
print(f"  Unit:        {rec.unit}")

# %% [2] Configure In-VRAM Pre-Processing Pipeline & 4×8 Planar Grid Layout
pipeline = Pipeline([
    BandpassFilter(low_hz=300.0, high_hz=3000.0, order=4, direction="forward-backward"),
    NotchFilter(freq_hz=60.0, q=30.0, direction="forward-backward"),
    CommonAverageReference(),
])

grid_rows, grid_cols = 4, 8
pitch_um = 100.0
grid_positions = [
    (float(c) * pitch_um, float(r) * pitch_um)
    for r in range(grid_rows)
    for c in range(grid_cols)
]
probe = syn.custom_layout("HDEMG-32-Grid", grid_positions[: rec.channels])

settling_l, settling_r = pipeline.settling(rec.sample_rate)
print(f"Pipeline settling halo at {rec.sample_rate:.2f} Hz: left={settling_l}, right={settling_r} samples")

# %% [3] Lazy Zero-Load Slicing & Out-of-Core CubeCL Streaming Sort
# `slice_time` creates a lightweight view without reading samples into RAM:
# - Use `rec.slice_time(start_sec=0.0, duration_sec=60.0)` for a 60 s window
# - Or pass `rec` directly to sort the entire recording
target_rec = rec.slice_time(start_sec=0.0, duration_sec=60.0)

threshold_factor = 5.0
pre_ms = 1.0
post_ms = 2.0

print(
    f"Streaming {target_rec} ({target_rec.samples:,} samples, {target_rec.duration_sec:.1f} s) "
    f"through Rust PrefetchReader + CubeCL..."
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
    batch_duration_sec=10.0,
    calibration_duration_sec=5.0,
    apply_sinc_shift=True,
)

templates = sort_res.all_templates()
spike_counts = sort_res.channel_spike_counts
active_channels = sum(1 for t in templates if t is not None)

print(f"\n[Out-of-Core Streaming Sort Summary]")
print(f"  Result:               {sort_res}")
print(f"  Dynamic Halos (L, R): {sort_res.halos} samples")
print(f"  Calibrated Mean σ_n:  {np.mean(sort_res.channel_sigmas_uv):.2f} {rec.unit}")
print(f"  Total Raw Crossings:  {sort_res.total_raw_crossings:,}")
print(f"  Total Deduped Spikes: {sort_res.total_dedup_spikes:,}")
print(f"  Active Grid Channels: {active_channels} / {rec.channels}")

# %% [4] Plot Welford-Accumulated Templates Across the 4×8 Electrode Grid
if HAS_PLT:
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
