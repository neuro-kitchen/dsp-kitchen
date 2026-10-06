# %% [markdown]
# # dsp-kitchen (`dsp-synapse` + `dsp-synapse-ml`): Large-Scale 1-Hour 32-Channel Diaphragm HD-EMG NWB Zarr Benchmark
# End-to-end evaluation of `dsp-synapse` out-of-core streaming and `dsp-synapse-ml` pretrained Kilosort4
# (`wTEMP` matched-filter detection + `wPCA` orthonormal basis denoising & projection) on the 1-hour (~19 GB)
# 32-channel Diaphragm HD-EMG NWB Zarr v3 recording (`data/nwb/15-25-33_meps.nwb.zarr`, `/acquisition/HDEMG`):
# 1. **Zero-Copy NWB Zarr v3 Inspection**: Opens `/acquisition/HDEMG` (32 channels, ~1 hour @ 24,414.0625 Hz, ~2.64B samples).
# 2. **Multi-Minute Out-of-Core Streaming Benchmark (`syn.sort_recording`)**:
#    - Streams multi-minute windows via Rust `PrefetchReader` + CubeCL GPU kernels in constant memory.
# 3. **Pretrained Kilosort4 (`wTEMP` + `wPCA`) Streaming Evaluation on HD-EMG**:
#    - Streams contiguous tiles through `Pipeline` (`BandpassFilter` 300–3000 Hz + 60 Hz `NotchFilter` + `CommonAverageReference`),
#      detects motor unit action potentials (MUAPs) via `Kilosort4Detector.from_hub()`, extracts sinc-aligned `[N, K=8, T=61]`
#      snippets across the 4×8 diaphragm grid, denoises via `Kilosort4BasisEmbedder.denoise()`, and projects into 48-D `wPCA` space.

# %% [1] Imports & Open the 1-Hour 32-Channel HD-EMG NWB Zarr v3 Recording
import time
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
from dsp_kitchen.synapse.ml import (
    Kilosort4BasisEmbedder,
    Kilosort4Detector,
    ModelHub,
)

nwb_path = dk.resolve_data_path("data/nwb/15-25-33_meps.nwb.zarr")
rec = dk.open_nwb_zarr(nwb_path, series="/acquisition/HDEMG")
fs = float(rec.sample_rate)
total_samples = rec.channels * rec.samples
raw_size_gb = (total_samples * 4) / (1024**3)

print(
    "=========================================================================================="
)
print(" 1-Hour Diaphragm HD-EMG NWB Zarr v3 Dataset (/acquisition/HDEMG)")
print(
    "=========================================================================================="
)
print(f"  Path:           {nwb_path}")
print(f"  Recording:      {rec.name} ({rec.series})")
print(
    f"  Shape:          {rec.shape} [channels={rec.channels}, samples={rec.samples:,}]"
)
print(f"  Sample Rate:    {fs:,.4f} Hz")
print(
    f"  Total Duration: {rec.duration_sec:,.2f} s ({rec.duration_sec / 60.0:.2f} min)"
)
print(f"  Uncompressed:   {raw_size_gb:.2f} GB (float32 equivalent)")
print(f"  Physical Unit:  {rec.unit}")
print(
    "=========================================================================================="
)

# %% [2] Configure 4×8 Diaphragm HD-EMG Grid & Pre-Processing Pipeline
grid_rows, grid_cols = 4, 8
pitch_um = 100.0
grid_positions = [
    (float(c) * pitch_um, float(r) * pitch_um)
    for r in range(grid_rows)
    for c in range(grid_cols)
]
probe = syn.custom_layout("Diaphragm-HDEMG-4x8", grid_positions[: rec.channels])

pipeline = Pipeline(
    [
        BandpassFilter(
            low_hz=300.0, high_hz=3000.0, order=4, direction="forward-backward"
        ),
        NotchFilter(freq_hz=60.0, q=30.0, direction="forward-backward"),
        CommonAverageReference(),
    ]
)

settling_l, settling_r = pipeline.settling(fs)
print(
    f"[Pipeline & Probe Configured] 4x8 HD-EMG Grid ({rec.channels} ch), settling halo = ({settling_l}, {settling_r}) samples"
)

# %% [3] Stage A: Multi-Minute Out-of-Core CubeCL Streaming Sort (`syn.sort_recording`)
# Stream a 5-minute (300 s = 7.32M samples/ch = 234.4M total samples) window out-of-core from Zarr v3
stream_duration_sec = min(300.0, float(rec.duration_sec))
target_slice = rec.slice_time(start_sec=0.0, duration_sec=stream_duration_sec)

t_stream_start = time.perf_counter()
sort_res = syn.sort_recording(
    target_slice,
    pipeline=pipeline,
    probe=probe,
    threshold_factor=5.0,
    refractory_ms=1.0,
    spatial_radius_um=150.0,
    k_neighbors=4,
    pre_ms=1.0,
    post_ms=2.0,
    batch_duration_sec=15.0,
    calibration_duration_sec=5.0,
    apply_sinc_shift=True,
)
stream_elapsed_sec = time.perf_counter() - t_stream_start
realtime_speedup = stream_duration_sec / stream_elapsed_sec

templates = sort_res.all_templates()
active_channels = sum(1 for t in templates if t is not None)

print(
    f"\n[Stage A: Out-of-Core Streaming Sort on {stream_duration_sec:.0f}s ({stream_duration_sec / 60.0:.1f} min) HD-EMG Window]"
)
print(
    f"  Wall-Clock Time:      {stream_elapsed_sec:.2f} s ({realtime_speedup:.1f}x real-time speedup)"
)
print(
    f"  Throughput:           {(target_slice.channels * target_slice.samples) / stream_elapsed_sec / 1e6:.2f} M samples/s"
)
print(f"  Calibrated Mean σ_n:  {np.mean(sort_res.channel_sigmas_uv):.4f} {rec.unit}")
print(f"  Total Raw Crossings:  {sort_res.total_raw_crossings:,}")
print(f"  Total Deduped MUAPs:  {sort_res.total_dedup_spikes:,}")
print(f"  Active Grid Channels: {active_channels} / {rec.channels}")

# %% [4] Stage B: Pretrained Kilosort4 (`wTEMP` + `wPCA`) Multi-Tile Streaming on HD-EMG
# Load verified Kilosort4 weights from the Model Hub in Rust
hub = ModelHub()
ks4_detector = Kilosort4Detector.from_hub(
    threshold_sigma=5.0, refractory_samples=int(0.001 * fs)
)
ks4_embedder = Kilosort4BasisEmbedder.from_hub()

# Stream 30 seconds (6 × 5.0 s tiles = 23.4M samples) across the recording to evaluate Kilosort4 wTEMP + wPCA on HD-EMG
ks4_eval_duration_sec = 30.0
tile_duration_sec = 5.0
num_tiles = int(ks4_eval_duration_sec // tile_duration_sec)
ks4_k = 8
ks4_nt = ks4_embedder.window_len  # 61 samples (~2.50 ms @ 24.4 kHz)

total_ks4_raw = 0
all_snippets = []
all_primary_ch = []
all_spike_times = []

t_ks4_start = time.perf_counter()
for tile_idx in range(num_tiles):
    t_start = tile_idx * tile_duration_sec
    chunk = rec.read_window(start_sec=t_start, duration_sec=tile_duration_sec)
    # Convert Volts to microvolts if stored in Volts so amplitudes are on standard uV scale
    if rec.unit.lower() in ("v", "volts", "volt"):
        chunk = chunk * 1e6
    filtered = pipeline.run(np.ascontiguousarray(chunk, dtype=np.float32), fs=fs)

    raw_events = ks4_detector.detect(filtered, sample_rate_hz=fs)
    total_ks4_raw += len(raw_events)

    dedup_events = syn.deduplicate_spikes(
        raw_events,
        probe,
        radius_um=150.0,
        window_samples=int(0.0008 * fs),
    )
    snippets_tile = syn.extract_snippets(
        filtered,
        dedup_events,
        probe,
        k_neighbors=ks4_k,
        pre_samples=20,
        post_samples=ks4_nt - 20,
        apply_sinc_shift=True,
    )
    if snippets_tile:
        tile_offset_samples = int(round(t_start * fs))
        all_snippets.append(
            np.stack([s.waveform() for s in snippets_tile], axis=0).astype(np.float32)
        )
        all_primary_ch.append(
            np.array([s.primary_channel for s in snippets_tile], dtype=np.int32)
        )
        all_spike_times.append(
            np.array(
                [int(s.center_sample) + tile_offset_samples for s in snippets_tile],
                dtype=np.int64,
            )
        )

snippets_nkt = np.concatenate(all_snippets, axis=0)
primary_channels = np.concatenate(all_primary_ch, axis=0)
spike_times_samples = np.concatenate(all_spike_times, axis=0)

# Denoise & embed all extracted 3D HD-EMG snippets in Rust via CubeCL Kilosort4 `wPCA`
denoised_nkt = ks4_embedder.denoise(snippets_nkt)
ks4_features = ks4_embedder.embed(snippets_nkt)
ks4_elapsed_sec = time.perf_counter() - t_ks4_start
ks4_speedup = ks4_eval_duration_sec / ks4_elapsed_sec

print(
    f"\n[Stage B: Rust Kilosort4 (`wTEMP` + `wPCA`) on {ks4_eval_duration_sec:.0f}s HD-EMG Stream]"
)
print(
    f"  Wall-Clock Time:         {ks4_elapsed_sec:.2f} s ({ks4_speedup:.1f}x real-time speedup)"
)
print(f"  wTEMP Crossings:         {total_ks4_raw:,}")
print(f"  Extracted MUAP Snippets: {snippets_nkt.shape} [N, K={ks4_k}, T={ks4_nt}]")
print(f"  wPCA Denoised Snippets:  {denoised_nkt.shape}")
print(f"  wPCA Projected Features: {ks4_features.shape} [N, K*6=48]")

# %% [5] Per-Channel Diaphragm HD-EMG Motor Unit Quality Metrics (Raw vs. `wPCA` Denoised)
unique_ch, counts = np.unique(primary_channels, return_counts=True)
top_channels = unique_ch[np.argsort(counts)[::-1][:6]]

print(
    "\n[Top Active Diaphragm HD-EMG Channels — Raw vs. Kilosort4 `wPCA` Denoised Metrics]"
)
print(
    f"  {'Grid Channel':<14s} | {'MUAPs':>7s} | {'Rate (Hz)':>9s} | {'Raw SNR':>9s} | {'wPCA Denoised SNR':>17s} | {'SNR Gain':>9s} | {'ISI Viol (%)':>12s}"
)
print("  " + "-" * 92)

for ch in top_channels:
    mask = primary_channels == ch
    ch_times = sorted(int(t) for t in spike_times_samples[mask])
    raw_primary = snippets_nkt[mask, 0, :]
    den_primary = denoised_nkt[mask, 0, :]

    raw_noise = float(np.std(raw_primary[:, :10])) + 1e-4
    den_noise = float(np.std(den_primary[:, :10])) + 1e-4
    raw_peak = float(np.min(raw_primary.mean(axis=0)))
    den_peak = float(np.min(den_primary.mean(axis=0)))

    snr_raw = syn.compute_snr(raw_peak, raw_noise)
    snr_den = syn.compute_snr(den_peak, den_noise)
    gain = snr_den / max(snr_raw, 1e-6)
    isi_stats = syn.compute_isi(
        ch_times,
        sample_rate_hz=fs,
        refractory_ms=1.5,
        total_duration_sec=ks4_eval_duration_sec,
    )
    ch_label = (
        rec.channel_names[int(ch)]
        if int(ch) < len(rec.channel_names)
        else f"Ch {int(ch)}"
    )

    print(
        f"  {ch_label:<14s} | {int(mask.sum()):7d} | {isi_stats['firing_rate_hz']:9.2f} | {snr_raw:9.2f} | {snr_den:17.2f} | {gain:8.2f}x | {isi_stats['violation_rate_pct']:11.2f}%"
    )

# %% [6] Plot HD-EMG Grid Activity, `wPCA` MUAP Denoising, and 48-D Feature Space
if HAS_PLT:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    t_ms = (np.arange(ks4_nt) - 20) / fs * 1000.0

    # Panel 1: 4x8 HD-EMG Grid Spike Count Heatmap
    grid_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for ch_idx, cnt in enumerate(
        sort_res.channel_spike_counts[: grid_rows * grid_cols]
    ):
        r, c = divmod(ch_idx, grid_cols)
        grid_map[r, c] = cnt
    im = axes[0].imshow(grid_map, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=axes[0], label="Detected MUAP Count (5 min)")
    axes[0].set_title("1. 4×8 Diaphragm HD-EMG Activity Map", fontweight="bold")
    axes[0].set_xlabel("Grid Column")
    axes[0].set_ylabel("Grid Row")

    # Panel 2: Raw vs. Kilosort4 `wPCA` Denoised MUAP Waveform on Top Channel
    best_ch = top_channels[0]
    best_idx = np.flatnonzero(primary_channels == best_ch)
    axes[1].plot(
        t_ms,
        snippets_nkt[best_idx[0], 0],
        color="#888888",
        alpha=0.8,
        linewidth=1.2,
        label=f"Raw MUAP #0 (Ch {best_ch})",
    )
    axes[1].plot(
        t_ms,
        denoised_nkt[best_idx[0], 0],
        color="#d62728",
        linewidth=2.0,
        label="wPCA Denoised MUAP #0",
    )
    axes[1].plot(
        t_ms,
        denoised_nkt[best_idx, 0].mean(axis=0),
        color="#1f77b4",
        linestyle="--",
        linewidth=1.8,
        label=f"Ch {best_ch} Mean Template (n={len(best_idx)})",
    )
    axes[1].axvline(0.0, color="#666666", linestyle=":", linewidth=0.9)
    axes[1].set_title("2. Kilosort4 wPCA MUAP Waveform Denoising", fontweight="bold")
    axes[1].set_xlabel("Time from Trough (ms)")
    axes[1].set_ylabel("Amplitude (μV)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Kilosort4 48-D `wPCA` Feature Projection (PC0 vs PC1)
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]
    for i, ch in enumerate(top_channels):
        mask = primary_channels == ch
        axes[2].scatter(
            ks4_features[mask, 0],
            ks4_features[mask, 1],
            s=18,
            alpha=0.78,
            color=colors[i % len(colors)],
            label=f"Ch {ch} (n={mask.sum()})",
        )
    axes[2].set_title("3. Kilosort4 wPCA Feature Space (PC0 vs PC1)", fontweight="bold")
    axes[2].set_xlabel("Primary Channel wPCA 0")
    axes[2].set_ylabel("Primary Channel wPCA 1")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %% [7] 4×8 Multi-Grid Plot of All 32 Detected HD-EMG Templates ± Standard Error (SE = SD / √n)
# Standard Error of the Mean: SE(t) = SD(t) / sqrt(n)
if HAS_PLT:
    fig_grid, grid_axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(18, 9.5),
        sharex=True,
        sharey=True,
    )
    t_ks4_ms = (np.arange(ks4_nt) - 20) / fs * 1000.0

    for ch in range(rec.channels):
        r, c = divmod(ch, grid_cols)
        ax = grid_axes[r, c]
        ch_name = (
            rec.channel_names[ch] if ch < len(rec.channel_names) else f"Ch {ch + 1}"
        )

        mask = primary_channels == ch
        n_ks4 = int(mask.sum())

        if n_ks4 > 0:
            raw_wf = snippets_nkt[mask, 0, :]
            den_wf = denoised_nkt[mask, 0, :]

            raw_mean = raw_wf.mean(axis=0)
            raw_sd = raw_wf.std(axis=0, ddof=1) if n_ks4 > 1 else np.zeros_like(raw_mean)
            raw_se = raw_sd / np.sqrt(n_ks4)

            den_mean = den_wf.mean(axis=0)
            den_sd = den_wf.std(axis=0, ddof=1) if n_ks4 > 1 else np.zeros_like(den_mean)
            den_se = den_sd / np.sqrt(n_ks4)

            # Plot ±1 SD (light background) and ±1 SE (95%/68% confidence of the mean template)
            ax.fill_between(
                t_ks4_ms,
                den_mean - den_sd,
                den_mean + den_sd,
                color="#1f77b4",
                alpha=0.12,
                label="±1 SD (wPCA)" if ch == 0 else None,
            )
            ax.fill_between(
                t_ks4_ms,
                raw_mean - raw_se,
                raw_mean + raw_se,
                color="#888888",
                alpha=0.35,
                label="Raw ±1 SE" if ch == 0 else None,
            )
            ax.plot(
                t_ks4_ms,
                raw_mean,
                color="#666666",
                linewidth=1.0,
                linestyle="--",
                label="Raw Mean" if ch == 0 else None,
            )
            ax.fill_between(
                t_ks4_ms,
                den_mean - den_se,
                den_mean + den_se,
                color="#d62728",
                alpha=0.45,
                label="wPCA ±1 SE" if ch == 0 else None,
            )
            ax.plot(
                t_ks4_ms,
                den_mean,
                color="#d62728",
                linewidth=1.5,
                label="wPCA Mean" if ch == 0 else None,
            )
            mean_se_uv = float(np.mean(den_se))
            ax.set_title(
                f"{ch_name} (n={n_ks4:,}, SE={mean_se_uv:.2f}μV)",
                fontsize=8,
                fontweight="bold",
            )
        else:
            ax.plot(
                t_ks4_ms,
                np.zeros_like(t_ks4_ms),
                color="#999999",
                linestyle="--",
                linewidth=0.8,
            )
            ax.set_title(f"{ch_name} (n=0)", fontsize=8)

        ax.axvline(0.0, color="#333333", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.axhline(0.0, color="#999999", linestyle="-", linewidth=0.5, alpha=0.4)
        ax.grid(True, linestyle="--", alpha=0.25)

        if r == grid_rows - 1:
            ax.set_xlabel("Time (ms)", fontsize=8)
        if c == 0:
            ax.set_ylabel(f"Amp ({rec.unit})", fontsize=8)

    handles, labels = grid_axes[0, 0].get_legend_handles_labels()
    if handles:
        fig_grid.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=5,
            fontsize=9,
            frameon=True,
        )

    fig_grid.suptitle(
        f"{rec.name} — 4×8 Diaphragm HD-EMG Detected MUAP Templates ± Standard Error (SE = SD / √n, Total MUAPs={len(snippets_nkt):,})",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    fig_out = dk.get_local_path() / "playground" / "figures" / "hdemg_4x8_templates_se.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig_grid.savefig(fig_out, dpi=160)
    print(f"\n[Saved 4x8 HD-EMG Multi-Grid Template ± SE Figure] -> {fig_out}")
    plt.show()

# %%
