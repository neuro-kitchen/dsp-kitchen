# %% [markdown]
# # dsp-kitchen (`dsp-synapse`): Simple Classical Spike Sorting Guide
# Step-by-step interactive notebook showing how to run classical in-memory spike sorting:
# 1. **Pre-processing Pipeline**: Bandpass (`300–3000 Hz`) + 60 Hz Notch + Spatial CAR
# 2. **Robust Noise Floor Estimation**: Quiroga Median Absolute Deviation (`syn.estimate_noise`, $\sigma_n = \text{median}(|x|)/0.6745$)
# 3. **Threshold Crossing Detection**: Multi-channel negative peak detection (`syn.detect_spikes`)
# 4. **Spatial-Temporal Deduplication**: Merging simultaneous crossings across neighboring sites (`syn.deduplicate_spikes`)
# 5. **Sub-Sample Sinc Realignment & $K$-NN Snippet Extraction**: `syn.extract_snippets(..., apply_sinc_shift=True)`
# 6. **Waveform PCA & Unit Quality Metrics**: `PCA`, `syn.compute_template`, `syn.compute_snr`, and `syn.compute_isi`

# %% [1] Imports & Load / Preprocess 32-Channel Recording Window
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
from dsp_kitchen.linalg import PCA

local_path = dk.get_local_path()
nwb_path = local_path / "playground" / "data" / "nwb" / "15-25-33_meps.nwb.zarr"

if nwb_path.exists():
    rec = dk.open_nwb_zarr(nwb_path, series="/acquisition/HDEMG")
    fs = rec.sample_rate
    duration_sec = 5.0
    raw = rec.read_window(start_sec=10.0, duration_sec=duration_sec)
else:
    fs = 30_000.0
    duration_sec = 2.0
    rng = np.random.default_rng(123)
    raw = rng.normal(0.0, 12.0, size=(32, int(fs * duration_sec))).astype(np.float32)

num_channels, num_samples = raw.shape

# Build a 4x8 planar electrode grid layout (100 um pitch)
grid_positions = [
    (float(c) * 100.0, float(r) * 100.0)
    for r in range(4)
    for c in range(8)
][:num_channels]
probe = syn.custom_layout("Planar-32-Grid", grid_positions)

pipeline = Pipeline([
    BandpassFilter(low_hz=300.0, high_hz=3000.0, order=4, direction="forward-backward"),
    NotchFilter(freq_hz=60.0, q=30.0, direction="forward-backward"),
    CommonAverageReference(),
])
filtered = pipeline.run(np.ascontiguousarray(raw, dtype=np.float32), fs=fs)

print(f"Filtered window: {filtered.shape} ({duration_sec:.1f} s @ {fs:.1f} Hz) on {probe.name}")

# %% [2] Step 1: Robust Noise Estimation (Quiroga MAD)
sigmas_uv = np.array([syn.estimate_noise(filtered[ch]) for ch in range(num_channels)], dtype=np.float32)

print("[Step 1: Quiroga MAD Noise Floor]")
print(f"  Mean sigma_n: {sigmas_uv.mean():.2f} uV (min={sigmas_uv.min():.2f}, max={sigmas_uv.max():.2f} uV)")

# %% [3] Step 2 & 3: Threshold Crossing Detection & Spatial Deduplication
threshold_factor = 4.5
refractory_samples = int(round(0.001 * fs))  # 1.0 ms refractory period
dedup_window_samples = int(round(0.0005 * fs))  # 0.5 ms spatial coincidence window

raw_spikes = syn.detect_spikes(
    filtered,
    threshold_factor=threshold_factor,
    refractory_samples=refractory_samples,
)

dedup_spikes = syn.deduplicate_spikes(
    raw_spikes,
    probe,
    radius_um=150.0,
    window_samples=dedup_window_samples,
)

reduction_pct = 100.0 * (1.0 - len(dedup_spikes) / max(1, len(raw_spikes)))
print("\n[Step 2 & 3: Detection & Spatial Deduplication]")
print(f"  Raw threshold crossings (-{threshold_factor} sigma): {len(raw_spikes):,}")
print(f"  Spatially deduplicated spikes:           {len(dedup_spikes):,} ({reduction_pct:.1f}% redundant copies removed)")
if dedup_spikes:
    print(f"  Example event: {dedup_spikes[0]}")

# %% [4] Step 4: Sub-Sample Sinc Realignment & Multi-Channel K-NN Snippet Extraction
k_neighbors = 4
pre_samples = int(round(0.001 * fs))   # 1.0 ms pre-trough
post_samples = int(round(0.002 * fs))  # 2.0 ms post-trough

snippets = syn.extract_snippets(
    filtered,
    dedup_spikes,
    probe,
    k_neighbors=k_neighbors,
    pre_samples=pre_samples,
    post_samples=post_samples,
    apply_sinc_shift=True,
)

print("\n[Step 4: Multi-Channel K-NN Snippet Extraction]")
print(f"  Extracted snippets: {len(snippets):,}")
if snippets:
    wf0 = snippets[0].waveform()
    print(f"  Snippet 0 shape:    {wf0.shape} [k_neighbors, num_samples], fractional shift={snippets[0].subsample_offset:+.3f} samples")

# %% [5] Step 5: Waveform PCA Feature Space & Simple Clustering
if len(snippets) >= 4:
    # Flatten each [K, T] snippet into a feature vector of length K * T, transposed to [features, n_spikes]
    flat_matrix = np.stack([s.waveform().reshape(-1) for s in snippets], axis=1).astype(np.float32)
    pca = PCA(n_components=3)
    pca.fit(flat_matrix)
    pc_scores = pca.transform(flat_matrix, use_gpu=True)  # [3, n_spikes]
    print("\n[Step 5: Waveform PCA Embedding]")
    print(f"  PC Scores Shape:          {pc_scores.shape} (components, spikes)")
    print(f"  Explained Variance Ratio: {np.round(pca.explained_variance_ratio, 4)}")

# %% [6] Step 6: Unit Template, SNR & ISI Violation Metrics
# Group spikes by their primary electrode channel and inspect the most active channel
counts_by_ch = {}
for idx, s in enumerate(snippets):
    counts_by_ch.setdefault(s.primary_channel, []).append(idx)

best_ch = max(counts_by_ch, key=lambda c: len(counts_by_ch[c])) if counts_by_ch else 0
ch_snippets = [snippets[i] for i in counts_by_ch.get(best_ch, [])]
ch_spike_samples = sorted(s.center_sample for s in ch_snippets)

template = syn.compute_template(ch_snippets) if ch_snippets else None
isi_stats = syn.compute_isi(
    ch_spike_samples,
    sample_rate_hz=fs,
    refractory_ms=1.5,
    total_duration_sec=duration_sec,
)

if template is not None:
    mean_wf = template["mean"]
    std_wf = template["std"]
    peak_uv = float(np.min(mean_wf[0]))
    snr = syn.compute_snr(peak_uv, float(sigmas_uv[best_ch]))
    print(f"\n[Step 6: Electrophysiology Quality Metrics (Primary Channel {best_ch})]")
    print(f"  Spike Count:          {len(ch_snippets)}")
    print(f"  Template Peak:        {peak_uv:.2f} uV (SNR = {snr:.2f})")
    print(f"  Firing Rate:          {isi_stats['firing_rate_hz']:.2f} Hz")
    print(f"  ISI Violations (<1.5ms): {isi_stats['violation_count']} ({isi_stats['violation_rate_pct']:.2f}%, ratio={isi_stats['isi_violations_ratio']:.4f})")

# %% [7] Diagnostic Visualization (Template + PCA Scatter + ISI Histogram)
if HAS_PLT and template is not None and len(snippets) >= 4:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Primary & neighbor waveforms on most active channel
    t_snip_ms = (np.arange(template["num_samples"]) - pre_samples) / fs * 1000.0
    for k in range(template["num_channels"]):
        lbl = f"Primary Ch {best_ch}" if k == 0 else f"Neighbor {k}"
        axes[0].plot(t_snip_ms, mean_wf[k], linewidth=1.8 if k == 0 else 1.0, label=lbl)
        if k == 0:
            axes[0].fill_between(t_snip_ms, mean_wf[0] - std_wf[0], mean_wf[0] + std_wf[0], alpha=0.2)
    axes[0].axvline(0.0, color="#d62728", linestyle=":", alpha=0.7)
    axes[0].set_title(f" Sinc-Aligned Template (Ch {best_ch}, n={len(ch_snippets)})", fontweight="bold")
    axes[0].set_xlabel("Time from Trough (ms)")
    axes[0].set_ylabel("Amplitude (μV)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: PC1 vs PC2 scatter colored by primary channel
    prim_chs = [s.primary_channel for s in snippets]
    sc = axes[1].scatter(pc_scores[0], pc_scores[1], c=prim_chs, cmap="tab20", s=16, alpha=0.8)
    plt.colorbar(sc, ax=axes[1], label="Primary Channel")
    axes[1].set_title("Waveform PCA Feature Space (PC1 vs. PC2)", fontweight="bold")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Inter-Spike Interval (ISI) distribution
    if len(ch_spike_samples) > 1:
        isis_ms = np.diff(ch_spike_samples) / fs * 1000.0
        axes[2].hist(isis_ms[isis_ms <= 50.0], bins=40, color="#2ca02c", alpha=0.8, edgecolor="white")
    axes[2].axvline(1.5, color="#d62728", linestyle="--", linewidth=1.2, label="1.5 ms Refractory Limit")
    axes[2].set_title(f"ISI Distribution (Ch {best_ch})", fontweight="bold")
    axes[2].set_xlabel("Inter-Spike Interval (ms)")
    axes[2].set_ylabel("Count")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
