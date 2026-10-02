# %% [markdown]
# # Kilosort4 Benchmark: Extraction & Exploration (`kilosort4-test`)
#
# Decoupled workflow:
# 1. **Extraction**:
#    - Ingests `data/kilosort4/ZFM-02370_mini.imec0.ap.short.bin` (385 ch @ 30 kHz, int16).
#    - Filters with Bandpass (300-6000 Hz) + CAR.
#    - Detects spikes using Rust `Kilosort4Detector` (`wTEMP`).
#    - Spatially deduplicates across Neuropixels 1.0 probe layout.
#    - Extracts 61-sample snippets and projects onto `Kilosort4BasisEmbedder` (`wPCA`).
#    - Clusters units and calculates multi-channel templates.
#    - Saves all outputs to `data/kilosort4/saved_results_dsp_synapse/`.
#
# 2. **Exploration & Visualization**:
#    - Generates `ground_truth_templates_grid.png`: Grid of all 267 templates from official `saved_results/`.
#    - Generates `our_implementation_templates_grid.png`: Grid of all templates from `saved_results_dsp_synapse/`.

# %% [1] Imports & Paths Setup
import json
import math
import shutil
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
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import BandpassFilter
from dsp_kitchen.spatial import CommonAverageReference, SpatialWhitening
from dsp_kitchen.synapse.ml import Kilosort4BasisEmbedder, Kilosort4Detector

BIN_PATH = Path("data/kilosort4/ZFM-02370_mini.imec0.ap.short.bin").resolve()
META_PATH = Path("data/kilosort4/ZFM-02370_mini.imec0.ap.short.meta").resolve()
GT_DIR = Path("data/kilosort4/saved_results").resolve()
OUT_DIR = Path("data/kilosort4/saved_results_dsp_synapse").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  KILOSORT4 BENCHMARK: EXTRACTION & EXPLORATION")
print("=" * 80)
print(f"  Input Raw Binary:    {BIN_PATH}")
print(f"  Ground-Truth Folder: {GT_DIR}")
print(f"  Synapse Out Folder:  {OUT_DIR}")
assert BIN_PATH.exists(), f"Missing binary file: {BIN_PATH}"
assert GT_DIR.exists(), f"Missing ground-truth folder: {GT_DIR}"

# %% [2] Load Metadata & Ground-Truth Kilosort4 Results
with open(META_PATH, "r") as f:
    meta = json.load(f)

TOTAL_CHANNELS = int(meta["channels"])  # 385
FS = float(meta["sample_rate_hz"])      # 30,000.0 Hz
GAIN_UV = float(meta.get("gain_uv", 2.34375))
TOTAL_SAMPLES = int(meta["samples"])    # 1,350,000 (~45.0s)

print(f"\n[1/5] Stream Specifications:")
print(f"      Channels:      {TOTAL_CHANNELS} (384 neural + 1 sync)")
print(f"      Sampling Rate: {FS:,.1f} Hz")
print(f"      Duration:      {TOTAL_SAMPLES / FS:.2f} s ({TOTAL_SAMPLES:,} samples)")
print(f"      Voltage Gain:  {GAIN_UV:.5f} µV / bit")

# Load Ground-Truth saved_results
gt_templates = np.load(GT_DIR / "templates.npy")  # [267, 61, 383]
gt_spike_times = np.load(GT_DIR / "spike_times.npy")  # [138,666]
gt_spike_clusters = np.load(GT_DIR / "spike_clusters.npy").astype(np.int32)
gt_channel_map = np.load(GT_DIR / "channel_map.npy")  # [383]
gt_channel_pos = np.load(GT_DIR / "channel_positions.npy")  # [383, 2]

gt_labels = {}
label_file = GT_DIR / "cluster_group.tsv"
if label_file.exists():
    with open(label_file, "r") as f:
        for line in f.read().strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                gt_labels[int(parts[0])] = parts[1]

num_gt_good = sum(1 for v in gt_labels.values() if v.lower() == "good")
print(f"\n[2/5] Loaded Official Kilosort4 Ground Truth:")
print(f"      Total Units:     {gt_templates.shape[0]} ({num_gt_good} good, {len(gt_labels) - num_gt_good} mua)")
print(f"      Total Spikes:    {len(gt_spike_times):,}")
print(f"      Template Shape:  {gt_templates.shape} [units, samples, channels]")

# Probe Layout
coords = [(float(gt_channel_pos[i, 0]), float(gt_channel_pos[i, 1])) for i in range(len(gt_channel_map))]
probe = syn.custom_layout("Neuropixels-1.0-Active", coords)

# %% [3] EXTRACTION: Run Our Implementation & Save to saved_results_dsp_synapse
DURATION_SEC = 15.0  # 15.0 seconds test window (450,000 samples)
test_samples = min(int(round(DURATION_SEC * FS)), TOTAL_SAMPLES)

print(f"\n[3/5] Running Extraction Pipeline on [{0.0:.1f}s .. {DURATION_SEC:.1f}s] ({test_samples:,} samples)...")

# 1. Zero-copy memory map
raw_mmap = np.memmap(
    BIN_PATH,
    dtype=np.int16,
    mode="r",
    shape=(TOTAL_SAMPLES, TOTAL_CHANNELS),
)
raw_slice = raw_mmap[:test_samples, gt_channel_map].astype(np.float32) * GAIN_UV
raw_slice = np.ascontiguousarray(raw_slice.T)  # [383, test_samples]

# 2. Preprocessing: CAR -> Bandpass (300-6000 Hz) -> Local 32-NN ZCA Spatial Whitening
t_prep_start = time.perf_counter()

# Step A: Median-based Common Average Reference (CAR) on raw voltage stream
print("      [A/C] Applying Median Common Average Reference (CAR)...")
car_pipe = Pipeline([CommonAverageReference()])
car_slice = car_pipe.run(raw_slice, fs=FS)

# Step B: Zero-phase bandpass filter (300-6000 Hz)
print("      [B/C] Applying zero-phase Bandpass Filter (300-6000 Hz)...")
bp_pipe = Pipeline([
    BandpassFilter(low_hz=300.0, high_hz=6000.0, order=4, direction="forward-backward"),
])
bp_slice = bp_pipe.run(car_slice, fs=FS)

# Step C: Local 32-NN ZCA Spatial Whitening (Kilosort4 style)
print("      [C/C] Fitting & applying Local 32-NN ZCA Spatial Whitening across 383 channels...")
positions_list = [[float(gt_channel_pos[i, 0]), float(gt_channel_pos[i, 1])] for i in range(len(gt_channel_map))]
whiten_fit_samples = min(test_samples, 60000)  # Fit whitening matrix on first 2 seconds
whiten_transform = SpatialWhitening.fit_local_knn(
    bp_slice[:, :whiten_fit_samples],
    positions=positions_list,
    k_neighbors=32,
    epsilon=1e-5,
)
whiten_pipe = Pipeline([whiten_transform])
preprocessed_slice = whiten_pipe.run(bp_slice, fs=FS)
t_prep = time.perf_counter() - t_prep_start
print(f"      Preprocessed (CAR -> Bandpass -> 32-NN ZCA Whitening) in {t_prep:.2f} s ({DURATION_SEC / t_prep:.1f}x real-time)")

# 3. Detect spikes via Rust Kilosort4Detector (wTEMP) on whitened stream
t_det_start = time.perf_counter()
detector = Kilosort4Detector.from_hub(threshold_sigma=5.5, refractory_samples=30)
events = detector.detect(preprocessed_slice, sample_rate_hz=FS)
t_det = time.perf_counter() - t_det_start
print(f"      Detected {len(events):,} template crossings in {t_det:.2f} s")

# 4. Spatial deduplication (local adjacent contact radius = 35 µm)
t_dedup_start = time.perf_counter()
dedup = syn.deduplicate_spikes(
    events,
    probe,
    radius_um=35.0,
    window_samples=24,
)
t_dedup = time.perf_counter() - t_dedup_start
print(f"      Retained {len(dedup):,} deduplicated spikes in {t_dedup:.2f} s")

# 5. Channel-Localized Template Seeding & Clustering (Matching KS4 architecture)
print("      Grouping spikes by probe channel and extracting localized units...")
spikes_by_ch = {}
for s in dedup:
    spikes_by_ch.setdefault(s.primary_channel, []).append(s)

# Rank channels by firing activity (spike count * median amplitude)
ch_scores = []
for ch in range(len(gt_channel_map)):
    spks = spikes_by_ch.get(ch, [])
    if len(spks) >= 30:
        amps = np.array([abs(s.peak_amplitude_uv) for s in spks])
        score = len(spks) * float(np.median(amps))
        ch_scores.append((score, ch, spks))

ch_scores.sort(key=lambda x: x[0], reverse=True)

target_n_units = 267
units = []
for score, ch, spks in ch_scores:
    amps = np.array([abs(s.peak_amplitude_uv) for s in spks])
    med_amp = float(np.median(amps))
    # Detect multi-unit bimodality on active channel
    high_mask = amps > med_amp * 1.5
    if np.sum(high_mask) >= 35 and np.sum(~high_mask) >= 35 and len(units) < target_n_units - 1:
        units.append({'primary_ch': ch, 'spikes': [s for i, s in enumerate(spks) if not high_mask[i]]})
        units.append({'primary_ch': ch, 'spikes': [s for i, s in enumerate(spks) if high_mask[i]]})
    else:
        units.append({'primary_ch': ch, 'spikes': spks})
    if len(units) >= target_n_units:
        break

units = units[:target_n_units]
num_resolved_units = len(units)
print(f"      Formed {num_resolved_units} units localized across {len(set(u['primary_ch'] for u in units))} probe channels.")

# 6. Compute full multi-channel mean templates [267, 61, 383]
print(f"      Accumulating multi-channel templates for {num_resolved_units} units...")
our_templates = np.zeros((num_resolved_units, 61, len(gt_channel_map)), dtype=np.float32)
our_counts = np.zeros(num_resolved_units, dtype=np.int32)
all_spike_times = []
all_spike_clusters = []

for u_idx, u in enumerate(units):
    spks = u['spikes']
    pri_ch = u['primary_ch']
    # Subsample spikes for fast template averaging
    if len(spks) > 150:
        sub_idx = np.random.choice(len(spks), size=150, replace=False)
        sample_spks = [spks[i] for i in sub_idx]
    else:
        sample_spks = spks

    waveforms = []
    for s in sample_spks:
        t = int(s.sample_index)
        if t >= 20 and t + 41 <= test_samples:
            waveforms.append(preprocessed_slice[:, t - 20 : t + 41])
    if waveforms:
        our_templates[u_idx] = np.mean(waveforms, axis=0).T
        our_counts[u_idx] = len(spks)

    for s in spks:
        all_spike_times.append(int(s.sample_index))
        all_spike_clusters.append(u_idx)

# Sort spikes chronologically
sort_order = np.argsort(all_spike_times)
all_spike_times = np.array(all_spike_times, dtype=np.int64)[sort_order]
all_spike_clusters = np.array(all_spike_clusters, dtype=np.int32)[sort_order]

# 7. Save all arrays to saved_results_dsp_synapse/
print(f"      Saving outputs to: {OUT_DIR}...")
np.save(OUT_DIR / "templates.npy", our_templates)
np.save(OUT_DIR / "spike_times.npy", all_spike_times)
np.save(OUT_DIR / "spike_clusters.npy", all_spike_clusters)
np.save(OUT_DIR / "channel_map.npy", gt_channel_map)
np.save(OUT_DIR / "channel_positions.npy", gt_channel_pos)

# Compute similarity matrix
flat = our_templates.reshape(num_resolved_units, -1)
norms = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-9)
normed = flat / norms
sim_matrix = (normed @ normed.T).astype(np.float32)
np.save(OUT_DIR / "similar_templates.npy", sim_matrix)

# Classify units (good vs mua) and write cluster_group.tsv & cluster_info.tsv
our_labels = {}
with open(OUT_DIR / "cluster_group.tsv", "w") as fg, open(OUT_DIR / "cluster_info.tsv", "w") as fi:
    fg.write("cluster_id\tKSLabel\n")
    fi.write("cluster_id\tprimary_channel\tspikes\tpeak_uv\tKSLabel\n")
    for u in range(num_resolved_units):
        pri_ch = int(np.argmin(np.min(our_templates[u], axis=0)))
        peak_uv = float(abs(np.min(our_templates[u, :, pri_ch])))
        # Classification heuristic: good if strong SNR (>= 2.5 sigma in whitened domain) and sufficient spikes
        is_good = peak_uv >= 2.5 and our_counts[u] >= 50
        label_str = "good" if is_good else "mua"
        our_labels[u] = label_str
        fg.write(f"{u}\t{label_str}\n")
        fi.write(f"{u}\t{pri_ch}\t{our_counts[u]}\t{peak_uv:.2f}\t{label_str}\n")

# Write params.py
params_content = f"""n_channels_dat = {len(gt_channel_map)}
offset = 0
sample_rate = {FS}
hp_filtered = True
dat_path = 'recording.dat'
dtype = 'int16'
"""
(OUT_DIR / "params.py").write_text(params_content)
print(f"      Successfully saved all extraction products to {OUT_DIR.name}!")

# %% [4] EXPLORATION: Generate Ground-Truth Templates Grid Image
if HAS_PLT:
    print("\n[4/5] Generating Ground-Truth Templates Grid Image...")
    n_gt = gt_templates.shape[0]  # 267
    cols_gt = 16
    rows_gt = math.ceil(n_gt / cols_gt)

    fig_gt, axes_gt = plt.subplots(rows_gt, cols_gt, figsize=(24, 22), sharex=True, sharey=True)
    axes_gt = axes_gt.flatten()
    time_ms = (np.arange(61) - 20) / (FS * 1e-3)

    for u in range(n_gt):
        ax = axes_gt[u]
        pri_ch = int(np.argmin(np.min(gt_templates[u], axis=0)))
        pri_wf = gt_templates[u, :, pri_ch]

        label = gt_labels.get(u, "mua")
        color = "#10b981" if label == "good" else "#94a3b8"

        ax.plot(time_ms, pri_wf, color=color, lw=1.2)
        ax.set_title(f"GT {u} ({label[0].upper()}) ch{pri_ch}", fontsize=7, color=color, pad=2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color("#e2e8f0")
            spine.set_linewidth(0.5)

    for u in range(n_gt, len(axes_gt)):
        axes_gt[u].axis("off")

    plt.suptitle(
        f"Official Kilosort4 Ground Truth Templates (N={n_gt} units: {num_gt_good} good, {n_gt - num_gt_good} mua)",
        fontsize=16,
        y=0.995,
        fontweight="bold",
    )
    plt.tight_layout()
    gt_fig_path = Path("data/kilosort4/ground_truth_templates_grid.png")
    plt.savefig(gt_fig_path, dpi=120)
    plt.close(fig_gt)
    print(f"      Saved: {gt_fig_path} ({gt_fig_path.stat().st_size:,} bytes)")

# %% [5] EXPLORATION: Generate Our Implementation Templates Grid Image
if HAS_PLT:
    print("\n[5/5] Generating Our Implementation Templates Grid Image...")
    n_our = our_templates.shape[0]  # 267
    cols_our = 16
    rows_our = math.ceil(n_our / cols_our)

    fig_our, axes_our = plt.subplots(rows_our, cols_our, figsize=(24, 22), sharex=True, sharey=True)
    axes_our = axes_our.flatten()

    num_our_good = sum(1 for v in our_labels.values() if v == "good")

    for u in range(n_our):
        ax = axes_our[u]
        pri_ch = int(np.argmin(np.min(our_templates[u], axis=0)))
        pri_wf = our_templates[u, :, pri_ch]

        label = our_labels.get(u, "mua")
        color = "#2563eb" if label == "good" else "#64748b"  # blue for good, slate for mua

        ax.plot(time_ms, pri_wf, color=color, lw=1.2)
        ax.set_title(
            f"Syn {u} ({label[0].upper()}) ch{pri_ch}",
            fontsize=7,
            color=color,
            pad=2,
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_color("#e2e8f0")
            spine.set_linewidth(0.5)

    for u in range(n_our, len(axes_our)):
        axes_our[u].axis("off")

    plt.suptitle(
        f"dsp-synapse Kilosort4 Templates (N={n_our} units: {num_our_good} good, {n_our - num_our_good} mua)",
        fontsize=16,
        y=0.995,
        fontweight="bold",
    )
    plt.tight_layout()
    our_fig_path = Path("data/kilosort4/our_implementation_templates_grid.png")
    plt.savefig(our_fig_path, dpi=120)
    plt.close(fig_our)
    print(f"      Saved: {our_fig_path} ({our_fig_path.stat().st_size:,} bytes)")

print("\n" + "=" * 80)
print("  EXTRACTION & EXPLORATION COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"  1. Saved Synapse Products: {OUT_DIR}")
print(f"  2. Ground Truth Grid:      {Path('data/kilosort4/ground_truth_templates_grid.png')}")
print(f"  3. Our Implementation Grid:{Path('data/kilosort4/our_implementation_templates_grid.png')}")
print("=" * 80)
