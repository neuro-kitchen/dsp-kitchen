# %% [markdown]
# # dsp-kitchen (`dsp-synapse`): Detection Against Ground Truth
# 1. A procedural recording with known spike times (`io.SyntheticRecording`: noise, power-line
#    hum, drifting units), or the same in streaming form
# 2. Pre-processing, noise, detection, spatial deduplication (as in `01_detection.py`)
# 3. Scoring with `syn.compare_spike_trains` (SpikeInterface-style matching within ±0.4 ms):
#    recall, precision, accuracy, for every unit and overall
#
# No data needed.

# %% [1] Imports & Ground-Truth Recording
import time
import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.io import SyntheticRecording
from dsp_kitchen.pipeline import Pipeline
from dsp_kitchen.filter.iir import HighpassFilter, NotchFilter

FS = 30_000.0
CHANNELS = 32
DURATION_SEC = 20.0
UNITS = 6
CONTACT_PITCH_UM = 25.0  # the synthetic units spread over neighbouring channels of a linear probe
LINE_HZ, NOTCH_Q = 60.0, 30.0
THRESHOLD_FACTOR = 5.0
REFRACTORY_SEC = 1e-3
DEDUP_RADIUS_UM = 4 * CONTACT_PITCH_UM
DEDUP_WINDOW_SEC = 0.5e-3

truth = SyntheticRecording(CHANNELS, FS, DURATION_SEC, units=UNITS)
rec = truth.recording()
probe = syn.ProbeLayout.from_positions("linear-32", [(0.0, i * CONTACT_PITCH_UM) for i in range(CHANNELS)])
gt_by_unit = [truth.spike_times(u) for u in range(truth.unit_count)]
gt_all = sorted(s for unit in gt_by_unit for s in unit)
print(f"{rec}: {truth.unit_count} units, {len(gt_all):,} ground-truth spikes")

# %% [2] Pre-processing, Detection & Deduplication
t0 = time.perf_counter()
raw = rec.read(0, rec.samples)
filtered = Pipeline([HighpassFilter(300.0), NotchFilter(LINE_HZ, NOTCH_Q)]).run(raw, fs=FS)
sigmas = syn.estimate_noise(filtered)
crossings = syn.detect_spikes(
    filtered, threshold_factor=THRESHOLD_FACTOR, refractory_samples=int(round(REFRACTORY_SEC * FS)), sigmas=list(sigmas)
)
spikes = syn.deduplicate_spikes(crossings, probe, radius_um=DEDUP_RADIUS_UM, window_samples=int(round(DEDUP_WINDOW_SEC * FS)))
elapsed_ms = (time.perf_counter() - t0) * 1000.0
detected = sorted(s.sample for s in spikes)
print(f"\n[Detection on {dk.runtime.current()}, {elapsed_ms:.0f} ms]")
print(f"  Mean noise σ:   {sigmas.mean():.2f} µV")
print(f"  Crossings:      {len(crossings):,}")
print(f"  After dedup:    {len(spikes):,}")

# %% [3] Scoring Against Ground Truth
overall = syn.compare_spike_trains(gt_all, detected, fs=FS)
print("\n[Overall (ground truth = A, detected = B)]")
print(f"  Recall {overall['recall']:.3f}, precision {overall['precision']:.3f}, accuracy {overall['accuracy']:.3f}")
print(f"  Missed {overall['false_negatives']:,}, extra {overall['false_positives']:,}")

# Per unit: recall only (a detected spike may belong to any unit)
per_unit = [syn.compare_spike_trains(times, detected, fs=FS)["recall"] for times in gt_by_unit]
for u, r in enumerate(per_unit):
    print(f"  unit {u}: {len(gt_by_unit[u]):5d} spikes, recall {r:.3f}")

# %% [4] Plot
if HAS_PLT:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["Recall", "Precision", "Accuracy"]
    values = [overall["recall"], overall["precision"], overall["accuracy"]]
    axes[0].bar(names, values, color=["#2ca02c", "#1f77b4", "#4c72b0"])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Detection vs ground truth (±0.4 ms)", fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar([f"unit {u}" for u in range(len(per_unit))], per_unit, color="#2ca02c")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Recall per unit", fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
