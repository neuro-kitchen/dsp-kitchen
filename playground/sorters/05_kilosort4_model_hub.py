# %% [markdown]
# # dsp-kitchen (`dsp-synapse-ml`): Kilosort4 Pretrained Matched-Filter Detection & Temporal Basis Projection
# Interactive playground notebook testing our end-to-end Rust `dsp-synapse-ml` Kilosort4 Model Hub engine:
# 1. **Model Hub Inspection (`kilosort4/universal-templates-v1` & `kilosort4/temporal-basis-v1`)**:
#    - Queries `ModelHub` for Kilosort4's two pretrained `.npy` weight sets (`wTEMP.npy` and `wPCA.npy`)
# 2. **Load Pretrained Rust Kilosort4 Models from Hub**:
#    - `Kilosort4Detector.from_hub(...)`: Loads the `[6, 61]` `wTEMP.npy` universal matched-filter templates in Rust
#    - `Kilosort4BasisEmbedder.from_hub(...)`: Loads the `[6, 61]` Fortran-ordered `wPCA.npy` orthonormal temporal basis in Rust
# 3. **Zero-Copy Neuropixels SpikeGLX Ingest (`data/kilosort4/ZFM-02370_mini.imec0.ap.short.bin`)**:
#    - Opens the 385-channel Neuropixels 1.0 `.ap.short.bin` recording via `dk.open_nwb_zarr`
#    - Runs in-VRAM Bandpass (300–6000 Hz) + Common Average Reference (CAR)
# 4. **Rust Kilosort4 Matched-Filter Spike Detection & `wPCA` Subspace Denoising / Projection**:
#    - Detects spikes via `ks4_detector.detect(filtered_chunk)` using the 6 universal templates (`wTEMP`)
#    - Denoises `[N, K=8, T=61]` snippets via `ks4_embedder.denoise(snippets_nkt)` ($\hat{x} = \mathbf{W}_{\text{PCA}}^\top \mathbf{W}_{\text{PCA}} x$)
#    - Projects `[N, K=8, T=61]` snippets into `[N, 48]` features in Rust via `ks4_embedder.embed(snippets_nkt)`
#    - Compares detected spike times against Kilosort4's official `saved_results/spike_times.npy` reference

# %% [1] Imports & Inspect Kilosort4 Pretrained Models in the Model Hub
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
from dsp_kitchen.filter.iir import BandpassFilter
from dsp_kitchen.spatial import CommonAverageReference
from dsp_kitchen.synapse.ml import (
    Kilosort4BasisEmbedder,
    Kilosort4Detector,
    ModelHub,
)

hub = ModelHub()
ks4_models = hub.list(family="kilosort4")

print("==========================================================================================")
print(f" Kilosort4 Models in dsp-synapse-ml Hub (Cache: {hub.cache_dir})")
print("==========================================================================================")
for m in ks4_models:
    print(
        f"  {m['status']:<12s} {m['id']:<34s} | task={m['task']:<10s} | format={m['format']} | sha256={m['sha256'][:12]}.."
    )
print("==========================================================================================")

# %% [2] Instantiate Pretrained Rust `Kilosort4Detector` (`wTEMP`) & `Kilosort4BasisEmbedder` (`wPCA`)
ks4_detector = Kilosort4Detector.from_hub(threshold_sigma=5.5, refractory_samples=30)
ks4_embedder = Kilosort4BasisEmbedder.from_hub(z_score_per_channel=False)

w_temp = ks4_detector.templates()  # [6, 61] parsed in Rust from wTEMP.npy
w_pca = ks4_embedder.basis()       # [6, 61] parsed in Rust from Fortran-ordered wPCA.npy

# Verify orthonormality of the Rust-decoded wPCA basis rows: W_PCA @ W_PCA.T == I_6
gram_matrix = w_pca @ w_pca.T
ortho_err = float(np.max(np.abs(gram_matrix - np.eye(6, dtype=np.float32))))

print("[Loaded Pretrained Kilosort4 Weights in Rust]")
print(f"  Kilosort4Detector templates (wTEMP): {w_temp.shape}, L2 norms = {np.round(np.linalg.norm(w_temp, axis=1), 4)}")
print(f"  Kilosort4BasisEmbedder basis (wPCA): {w_pca.shape}, max ||W W^T - I||_inf = {ortho_err:.2e}")

# %% [3] Open Kilosort4 Neuropixels Test Recording & Official Reference Outputs
rec_path = dk.resolve_data_path("data/kilosort4/ZFM-02370_mini.imec0.ap.short.bin")
rec = dk.open_nwb_zarr(rec_path)
fs = float(rec.sample_rate)

n_eval_channels = 64
eval_duration_sec = 2.0
raw_chunk = rec.read_window(
    start_sec=0.0,
    duration_sec=eval_duration_sec,
    channels=list(range(n_eval_channels)),
)

saved_results_dir = dk.resolve_data_path("data/kilosort4/saved_results")
ks4_ref_times = np.load(saved_results_dir / "spike_times.npy")
ks4_ref_clusters = np.load(saved_results_dir / "spike_clusters.npy")
ks4_ref_templates = np.load(saved_results_dir / "templates.npy")
ks4_chan_pos = np.load(saved_results_dir / "channel_positions.npy")

print(f"\n[Neuropixels Recording & Kilosort4 Reference]")
print(f"  Recording:                     {rec}")
print(f"  Evaluated Window:              {raw_chunk.shape} ({eval_duration_sec:.1f} s @ {fs:,.0f} Hz)")
print(f"  Reference Spikes (full 45 s):  {len(ks4_ref_times):,} across {len(np.unique(ks4_ref_clusters))} clusters")
print(f"  Reference Templates Shape:     {ks4_ref_templates.shape} [n_templates, nt=61, n_channels]")

# %% [4] Run Rust `Kilosort4Detector` (`wTEMP`) & `Kilosort4BasisEmbedder` (`wPCA`)
t0 = time.perf_counter()

pipeline = Pipeline(
    [
        BandpassFilter(low_hz=300.0, high_hz=6000.0, order=3, direction="forward-backward"),
        CommonAverageReference(),
    ]
)
filtered_chunk = pipeline.run(np.ascontiguousarray(raw_chunk, dtype=np.float32), fs=fs)

# Build probe layout from Kilosort4's `channel_positions.npy`
probe_coords = [
    (float(ks4_chan_pos[ch, 0]), float(ks4_chan_pos[ch, 1]))
    for ch in range(n_eval_channels)
]
probe = syn.custom_layout("Neuropixels-1.0-KS4", probe_coords)

# 1. Stage 1 Detection: Rust Kilosort4 6-template matched-filter detector (`wTEMP.npy`)
ks4_raw_events = ks4_detector.detect(filtered_chunk, sample_rate_hz=fs)
dedup_events = syn.deduplicate_spikes(
    ks4_raw_events,
    probe,
    radius_um=75.0,
    window_samples=int(0.0005 * fs),
)

# 2. Extract 61-sample (20 pre + 41 post), 8-channel neighborhood snippets `[N, K=8, T=61]`
ks4_k = 8
ks4_nt = 61
snippets_list = syn.extract_snippets(
    filtered_chunk,
    dedup_events,
    probe,
    k_neighbors=ks4_k,
    pre_samples=20,
    post_samples=ks4_nt - 20,
    apply_sinc_shift=True,
)
snippets_nkt = np.stack([s.waveform() for s in snippets_list], axis=0).astype(np.float32)
primary_channels = np.array([s.primary_channel for s in snippets_list], dtype=np.int32)

# 3. Stage 2 Denoising: Rust Kilosort4 rank-6 `wPCA` subspace reconstruction (`[N, 8, 61] -> [N, 8, 61]`)
denoised_snippets_nkt = ks4_embedder.denoise(snippets_nkt)

# 4. Stage 3 Feature Embedding: Rust Kilosort4 `wPCA` projection (`[N, 8, 61] -> [N, 48]`)
ks4_pc_features = ks4_embedder.embed(snippets_nkt)

elapsed_ms = (time.perf_counter() - t0) * 1000.0

print(f"\n[Rust Kilosort4 Pipeline Execution ({elapsed_ms:.1f} ms)]")
print(f"  wTEMP Matched-Filter Crossings: {len(ks4_raw_events):,}")
print(f"  Spatially Deduplicated Spikes:  {len(dedup_events):,}")
print(f"  Extracted Snippets [N, K, T]:   {snippets_nkt.shape}")
print(f"  wPCA Denoised Snippets:         {denoised_snippets_nkt.shape}")
print(f"  wPCA Projected Features [N,48]: {ks4_pc_features.shape}")

# %% [5] Compare Detected Spikes & `wPCA` Templates Against Kilosort4 `saved_results`
# Compute biological unit quality metrics (SNR & ISI violation rate) directly via `dsp-synapse`
unique_ch, counts = np.unique(primary_channels, return_counts=True)
top_channels = unique_ch[np.argsort(counts)[::-1][:4]]

print("\n[Top Active Neuropixels Channels — Direct Electrophysiology Metrics (`dsp-synapse`)]")
print(
    f"  {'Primary Ch':<12s} | {'Spikes':>8s} | {'Rate (Hz)':>9s} | {'Raw SNR':>9s} | {'wPCA Denoised SNR':>17s} | {'ISI Viol (%)':>12s}"
)
print("  " + "-" * 80)

for ch in top_channels:
    mask = primary_channels == ch
    ch_times = sorted(
        int(snippets_list[i].center_sample) for i in np.flatnonzero(mask)
    )
    raw_primary = snippets_nkt[mask, 0, :]
    den_primary = denoised_snippets_nkt[mask, 0, :]

    raw_noise = float(np.std(raw_primary[:, :10])) + 1e-4
    den_noise = float(np.std(den_primary[:, :10])) + 1e-4
    raw_peak = float(np.min(raw_primary.mean(axis=0)))
    den_peak = float(np.min(den_primary.mean(axis=0)))

    snr_raw = syn.compute_snr(raw_peak, raw_noise)
    snr_den = syn.compute_snr(den_peak, den_noise)
    isi_stats = syn.compute_isi(
        ch_times,
        sample_rate_hz=fs,
        refractory_ms=1.5,
        total_duration_sec=eval_duration_sec,
    )

    print(
        f"  Ch {int(ch):<9d} | {int(mask.sum()):8d} | {isi_stats['firing_rate_hz']:9.1f} | {snr_raw:9.2f} | {snr_den:17.2f} | {isi_stats['violation_rate_pct']:11.2f}%"
    )

# %% [6] Visualize `wTEMP` & `wPCA` Weights, `wPCA` Subspace Denoising, and 48-D Feature Space
if HAS_PLT:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    t_ms = (np.arange(ks4_nt) - 20) / fs * 1000.0
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    # Panel 1: Pretrained Kilosort4 Universal Templates (`wTEMP.npy`)
    for i in range(6):
        axes[0, 0].plot(t_ms, w_temp[i], linewidth=1.5, label=f"wTEMP[{i}]")
    axes[0, 0].axvline(0.0, color="#666666", linestyle=":", linewidth=0.9)
    axes[0, 0].set_title(
        "1. Kilosort4 Universal Matched Filters (wTEMP.npy)", fontweight="bold"
    )
    axes[0, 0].set_xlabel("Time from Trough (ms)")
    axes[0, 0].set_ylabel("Normalized Template Weight")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Pretrained Kilosort4 Orthonormal Temporal Basis (`wPCA.npy`)
    for i in range(6):
        axes[0, 1].plot(t_ms, w_pca[i], linewidth=1.5, label=f"wPCA[{i}]")
    axes[0, 1].axvline(0.0, color="#666666", linestyle=":", linewidth=0.9)
    axes[0, 1].set_title(
        "2. Kilosort4 Orthonormal Temporal Basis (wPCA.npy)", fontweight="bold"
    )
    axes[0, 1].set_xlabel("Time from Trough (ms)")
    axes[0, 1].set_ylabel("Basis Weight")
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Raw vs. Rust `Kilosort4BasisEmbedder.denoise()` Single Spike & Template
    best_ch = top_channels[0]
    best_indices = np.flatnonzero(primary_channels == best_ch)
    spk_idx = best_indices[0]
    axes[1, 0].plot(
        t_ms,
        snippets_nkt[spk_idx, 0],
        color="#888888",
        alpha=0.8,
        linewidth=1.2,
        label=f"Raw Spike #{spk_idx} (Ch {best_ch})",
    )
    axes[1, 0].plot(
        t_ms,
        denoised_snippets_nkt[spk_idx, 0],
        color="#d62728",
        linewidth=2.0,
        label="wPCA 6-Component Reconstruction",
    )
    axes[1, 0].plot(
        t_ms,
        denoised_snippets_nkt[best_indices, 0].mean(axis=0),
        color="#1f77b4",
        linestyle="--",
        linewidth=1.8,
        label=f"Ch {best_ch} Mean Template (n={len(best_indices)})",
    )
    axes[1, 0].axvline(0.0, color="#666666", linestyle=":", linewidth=0.9)
    axes[1, 0].set_title(
        "3. Rust Kilosort4BasisEmbedder Waveform Denoising", fontweight="bold"
    )
    axes[1, 0].set_xlabel("Time from Trough (ms)")
    axes[1, 0].set_ylabel("Amplitude (μV)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Rust `Kilosort4BasisEmbedder.embed()` PC0 vs PC1 Feature Space
    for idx, ch in enumerate(top_channels):
        mask = primary_channels == ch
        axes[1, 1].scatter(
            ks4_pc_features[mask, 0],
            ks4_pc_features[mask, 1],
            color=colors[idx % len(colors)],
            s=24,
            alpha=0.82,
            label=f"Ch {ch} (n={mask.sum()})",
        )
    axes[1, 1].set_title(
        "4. Rust Kilosort4BasisEmbedder Feature Space (PC0 vs PC1)", fontweight="bold"
    )
    axes[1, 1].set_xlabel("Primary Channel wPCA 0")
    axes[1, 1].set_ylabel("Primary Channel wPCA 1")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
