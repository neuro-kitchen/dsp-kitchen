# %% [markdown]
# # dsp-kitchen (`dsp-synapse`): Ground-Truth Spike Sorting Evaluation & Scorecard
# Interactive notebook version of the ground-truth evaluation suite:
# 1. Loads the 32-channel MEArec ground-truth recording (or generates a calibrated 32-channel synthetic ground-truth benchmark if MEArec files are absent)
# 2. Runs Quiroga MAD noise estimation, threshold detection, spatial deduplication, sub-sample sinc realignment, and $K$-NN snippet extraction
# 3. Computes ground-truth **Sensitivity (Recall)**, **Precision**, **False Discovery Rate (FDR)**, **Accuracy ($TP / (TP + FP + FN)$)**, and **Sub-sample Timing Jitter**

# %% [1] Imports & Load Ground-Truth Dataset (MEArec or Synthetic Ground Truth)
import json
import time
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.linalg import PCA

local_path = dk.get_local_path()
data_dir = local_path / "playground" / "data"
bin_path = data_dir / "mearec_32ch_10s.bin"
meta_path = data_dir / "mearec_32ch_10s.meta"
gt_path = data_dir / "mearec_32ch_ground_truth.json"

if bin_path.exists() and meta_path.exists() and gt_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)
    num_channels = meta["channels"]
    num_samples = meta["samples"]
    fs = float(meta["sample_rate_hz"])
    duration = float(meta["duration_seconds"])
    raw_data = np.fromfile(bin_path, dtype=np.float32).reshape(
        (num_channels, num_samples)
    )
    probe_coords = [(0.0, float(i * 25.0)) for i in range(num_channels)]
    probe = syn.custom_layout("Neuronexus-32", probe_coords)
    gt_spikes = sorted(
        (int(s), int(uid))
        for uid, udata in gt_data.items()
        for s in udata["sample_indices"]
    )
    dataset_label = "MEArec 32-ch Ground Truth"
else:
    # Synthesize a 32-channel, 5-second ground-truth recording with 4 well-separated units
    fs = 30_000.0
    duration = 5.0
    num_channels = 32
    num_samples = int(fs * duration)
    rng = np.random.default_rng(99)
    raw_data = rng.normal(0.0, 10.0, size=(num_channels, num_samples)).astype(
        np.float32
    )
    probe_coords = [
        (float((i % 2) * 25.0), float((i // 2) * 25.0)) for i in range(num_channels)
    ]
    probe = syn.custom_layout("Staggered-32", probe_coords)

    t_wf = np.linspace(-0.001, 0.002, 90, dtype=np.float32)
    trough_offset = int(np.argmin(-np.exp(-((t_wf / 0.00022) ** 2))))
    unit_centers = [4, 11, 19, 27]
    gt_spikes = []
    for uid, center_ch in enumerate(unit_centers):
        n_events = 60
        isi_samples = rng.integers(int(0.015 * fs), int(0.120 * fs), size=n_events)
        event_samples = np.cumsum(isi_samples)
        event_samples = event_samples[event_samples + 90 < num_samples - 100]
        wf = (-135.0 - 15.0 * uid) * np.exp(-((t_wf / 0.00022) ** 2)) + 45.0 * np.exp(
            -(((t_wf - 0.00045) / 0.00035) ** 2)
        )
        for s in event_samples:
            for ch in range(max(0, center_ch - 2), min(num_channels, center_ch + 3)):
                atten = np.exp(-0.45 * abs(ch - center_ch))
                raw_data[ch, s : s + 90] += (atten * wf).astype(np.float32)
            gt_spikes.append((int(s + trough_offset), uid))
    gt_spikes.sort(key=lambda x: x[0])
    dataset_label = "Synthetic 32-ch Ground Truth (4 Units)"

print(f"[{dataset_label}]")
print(f"  Shape:            {raw_data.shape} ({duration:.1f} s @ {fs:,.0f} Hz)")
print(f"  Ground-Truth:     {len(gt_spikes)} total spikes")

# %% [2] Run Detection, Spatial Deduplication & Sinc Snippet Extraction
t0 = time.perf_counter()
sigmas = np.array([syn.estimate_noise(raw_data[ch]) for ch in range(num_channels)])
raw_crossings = syn.detect_spikes(
    raw_data,
    threshold_factor=5.0,
    refractory_samples=int(0.0008 * fs),
)
dedup_spikes = syn.deduplicate_spikes(
    raw_crossings,
    probe,
    radius_um=75.0,
    window_samples=int(0.0005 * fs),
)
snippets = syn.extract_snippets(
    raw_data,
    dedup_spikes,
    probe,
    k_neighbors=7,
    pre_samples=int(0.001 * fs),
    post_samples=int(0.002 * fs),
    apply_sinc_shift=True,
)
elapsed_ms = (time.perf_counter() - t0) * 1000.0

print(f"\n[Sorting Execution ({elapsed_ms:.1f} ms)]")
print(f"  Mean Noise Floor: {sigmas.mean():.2f} uV")
print(f"  Raw Crossings:    {len(raw_crossings):,}")
print(f"  Deduped Spikes:   {len(dedup_spikes):,}")
print(f"  Snippets:         {len(snippets):,}")

# %% [3] Ground-Truth Matching & Accuracy Scorecard
tol_samples = int(round(0.001 * fs))  # ±1.0 ms matching tolerance
det_times = np.array([int(d.sample_index) for d in dedup_spikes], dtype=np.int64)
gt_times = np.array([g[0] for g in gt_spikes], dtype=np.int64)

matched_det = set()
jitter_samples = []
tp = 0

for gt_t in gt_times:
    if len(det_times) == 0:
        break
    diffs = np.abs(det_times - gt_t)
    best_idx = int(np.argmin(diffs))
    if diffs[best_idx] <= tol_samples and best_idx not in matched_det:
        matched_det.add(best_idx)
        tp += 1
        jitter_samples.append(int(det_times[best_idx] - gt_t))

fn = len(gt_times) - tp
fp = len(det_times) - tp
recall = tp / max(1, tp + fn)
precision = tp / max(1, tp + fp)
fdr = fp / max(1, tp + fp)
accuracy = tp / max(1, tp + fp + fn)
jitter_ms = np.std(jitter_samples) / fs * 1000.0 if jitter_samples else 0.0

print("\n[Ground-Truth Accuracy Scorecard (±1.0 ms tolerance)]")
print(f"  True Positives (TP):   {tp}")
print(f"  False Negatives (FN):  {fn}")
print(f"  False Positives (FP):  {fp}")
print(f"  Sensitivity (Recall):  {recall * 100:6.2f}%")
print(f"  Precision:             {precision * 100:6.2f}%")
print(f"  False Discovery Rate:  {fdr * 100:6.2f}%")
print(f"  Overall Accuracy:      {accuracy * 100:6.2f}%")
print(f"  Timing Jitter (σ):     {jitter_ms:6.3f} ms")

# %% [4] Plot Accuracy Scorecard & Timing Error Distribution
if HAS_PLT:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    metrics_names = ["Recall", "Precision", "Accuracy", "1 - FDR"]
    metrics_vals = [
        recall * 100.0,
        precision * 100.0,
        accuracy * 100.0,
        (1.0 - fdr) * 100.0,
    ]
    bars = axes[0].bar(
        metrics_names, metrics_vals, color=["#2ca02c", "#1f77b4", "#4c72b0", "#55a868"]
    )
    axes[0].set_ylim(0.0, 108.0)
    for bar, v in zip(bars, metrics_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 1.5,
            f"{v:.1f}%",
            ha="center",
            fontweight="bold",
        )
    axes[0].set_title(f"{dataset_label} — Scorecard", fontweight="bold")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].grid(True, axis="y", alpha=0.3)

    if jitter_samples:
        jitter_arr_ms = np.array(jitter_samples) / fs * 1000.0
        axes[1].hist(
            jitter_arr_ms, bins=21, color="#4c72b0", edgecolor="white", alpha=0.85
        )
    axes[1].axvline(0.0, color="#d62728", linestyle="--", linewidth=1.2)
    axes[1].set_title(
        f"Detection Timing Error (σ = {jitter_ms:.3f} ms)", fontweight="bold"
    )
    axes[1].set_xlabel("Detected − True Spike Time (ms)")
    axes[1].set_ylabel("Matched Spikes")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
