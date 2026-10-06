# %% [markdown]
# # dsp-kitchen (`dsp-synapse`): Spike Detection Walkthrough
# Detection on a window held in memory (no clustering: spikes are grouped by primary channel):
# 1. **Pre-processing pipeline**: band-pass (300–3000 Hz) + notch + common reference, on the device
# 2. **Noise per channel**: median absolute deviation (`syn.estimate_noise`, σ = median(|x|) / 0.6745)
# 3. **Threshold crossings**: local extrema beyond k·σ (`syn.detect_spikes`)
# 4. **Spatial deduplication**: one spike per action potential across neighbouring sites
# 5. **Snippets**: nearest channels, realigned to the sub-sample trough (`syn.extract_snippets`)
# 6. **PCA, template, SNR, ISI violations** on the busiest channel
#
# Data (local, git-ignored): `data/nwb/15-25-33_meps.nwb.zarr` (see playground/README.md) (HD-EMG, 4 × 8 grid);
# synthetic noise is used when it is absent.

# %% [1] Imports, Recording & Pre-processing
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
from dsp_kitchen.linalg import PCA

# Test data lives outside the playground: `<repository>/data/`, or the folder in DSP_KITCHEN_DATA
DATA_DIR = Path(os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / "data"))
NWB_PATH = DATA_DIR / "nwb" / "15-25-33_meps.nwb.zarr"
GRID_ROWS, GRID_COLS = 4, 8
ELECTRODE_PITCH_UM = 100.0  # inter-electrode distance of the grid (set your array's)
WINDOW = (10.0, 5.0)  # (start, duration) in seconds
LINE_HZ, NOTCH_Q = 60.0, 30.0

if NWB_PATH.exists():
    hdemg = next((s["id"] for s in list_sources(str(NWB_PATH)) if "HDEMG" in s["id"]), None)
    rec = Recording(str(NWB_PATH), source=hdemg)
    fs, unit = rec.sample_rate, rec.units[0]
    raw = rec.read_window(*WINDOW)
else:
    fs, unit = 30_000.0, ""
    raw = np.random.default_rng(123).standard_normal((GRID_ROWS * GRID_COLS, int(fs * 2.0))).astype(np.float32)

num_channels, num_samples = raw.shape
duration_sec = num_samples / fs
probe = syn.ProbeLayout.hdemg_grid("HD-EMG 4x8", GRID_ROWS, GRID_COLS, ELECTRODE_PITCH_UM)

pipeline = Pipeline([BandpassFilter(300.0, 3000.0, order=4), NotchFilter(LINE_HZ, NOTCH_Q), CommonAverageReference()])
filtered = pipeline.run(raw, fs=fs)
print(f"Filtered window: {filtered.shape} ({duration_sec:.1f} s @ {fs:.1f} Hz, {unit or 'a.u.'}) on {probe.name}")

# %% [2] Step 1: Robust Noise Estimation (Quiroga MAD)
sigmas = syn.estimate_noise(filtered)  # one σ per channel

print("[Step 1: Noise per channel (MAD)]")
print(f"  Mean σ: {sigmas.mean():.3g} {unit} (min={sigmas.min():.3g}, max={sigmas.max():.3g})")

# %% [3] Step 2 & 3: Threshold Crossing Detection & Spatial Deduplication
THRESHOLD_FACTOR = 4.5
REFRACTORY_SEC = 1e-3  # between spikes on one channel
DEDUP_WINDOW_SEC = 0.5e-3  # coincidence window across channels
DEDUP_RADIUS_UM = 1.5 * ELECTRODE_PITCH_UM  # direct and diagonal neighbours
refractory_samples = int(round(REFRACTORY_SEC * fs))
dedup_window_samples = int(round(DEDUP_WINDOW_SEC * fs))

raw_spikes = syn.detect_spikes(
    filtered,
    threshold_factor=THRESHOLD_FACTOR,
    refractory_samples=refractory_samples,
    sigmas=list(sigmas),
)

dedup_spikes = syn.deduplicate_spikes(
    raw_spikes,
    probe,
    radius_um=DEDUP_RADIUS_UM,
    window_samples=dedup_window_samples,
)

reduction_pct = 100.0 * (1.0 - len(dedup_spikes) / max(1, len(raw_spikes)))
print("\n[Step 2 & 3: Detection & Spatial Deduplication]")
print(f"  Threshold crossings (-{THRESHOLD_FACTOR} σ): {len(raw_spikes):,}")
print(f"  Spatially deduplicated spikes:           {len(dedup_spikes):,} ({reduction_pct:.1f}% redundant copies removed)")
if dedup_spikes:
    print(f"  Example event: {dedup_spikes[0]}")

# %% [4] Step 4: Sub-Sample Sinc Realignment & Multi-Channel K-NN Snippet Extraction
K_NEIGHBORS = 4
PRE_SEC, POST_SEC = 1e-3, 2e-3  # before / after the trough
k_neighbors = K_NEIGHBORS
pre_samples = int(round(PRE_SEC * fs))
post_samples = int(round(POST_SEC * fs))

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
    pc_scores = pca.transform(flat_matrix)  # [3, n_spikes], on the current runtime
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
isi_stats = syn.compute_isi(ch_spike_samples, fs=fs, duration_sec=duration_sec)  # 1.5 ms threshold (SpikeInterface)

if template is not None:
    mean_wf = template["mean"]
    std_wf = template["std"]
    peak = float(np.min(mean_wf[0]))
    snr = syn.compute_snr(abs(peak), float(sigmas[best_ch]))
    print(f"\n[Step 6: Electrophysiology Quality Metrics (Primary Channel {best_ch})]")
    print(f"  Spike Count:          {len(ch_snippets)}")
    print(f"  Template Peak:        {peak:.3g} {unit} (SNR = {snr:.2f})")
    print(f"  Firing Rate:          {isi_stats['firing_rate_hz']:.2f} Hz")
    print(f"  ISI Violations (<1.5ms): {isi_stats['violation_count']} ({isi_stats['violation_rate_pct']:.2f}%, ratio={isi_stats['isi_violations_ratio']:.4f})")

# %% [7] Diagnostic Visualization (Template + PCA Scatter + ISI Histogram)
if HAS_PLT and template is not None and len(snippets) >= 4:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Primary & neighbor waveforms on most active channel
    t_snip_ms = (np.arange(mean_wf.shape[1]) - pre_samples) / fs * 1000.0
    for k in range(mean_wf.shape[0]):
        lbl = f"Primary Ch {best_ch}" if k == 0 else f"Neighbor {k}"
        axes[0].plot(t_snip_ms, mean_wf[k], linewidth=1.8 if k == 0 else 1.0, label=lbl)
        if k == 0:
            axes[0].fill_between(t_snip_ms, mean_wf[0] - std_wf[0], mean_wf[0] + std_wf[0], alpha=0.2)
    axes[0].axvline(0.0, color="#d62728", linestyle=":", alpha=0.7)
    axes[0].set_title(f" Sinc-Aligned Template (Ch {best_ch}, n={len(ch_snippets)})", fontweight="bold")
    axes[0].set_xlabel("Time from Trough (ms)")
    axes[0].set_ylabel(f"Amplitude ({unit or 'a.u.'})")
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
