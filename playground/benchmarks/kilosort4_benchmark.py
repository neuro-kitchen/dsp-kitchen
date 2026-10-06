# %% [markdown]
# # dsp-kitchen benchmark: Kilosort4 speed and outputs vs Kilosort4's own results
# Runs our Kilosort4 (preprocessing fit, universal templates, universal-template detection) over
# the Neuropixels test recording and compares it with what Kilosort4 recorded in
# `data/kilosort4/saved_results/`:
# 1. **Speed**: wall time per stage on each chosen runtime (first run includes kernel
#    compilation; later runs are warm), as a multiple of real time; Kilosort4's recorded total.
# 2. **Whitening**: our matrix vs `whitening_mat_dat.npy` (same channels, same order), up to a
#    scale: Kilosort4 whitens the stored integers, we whiten values in their unit (µV).
# 3. **Spikes**: our detections vs Kilosort4's final spikes (recall, precision within ±0.4 ms),
#    per Kilosort4 unit, and the vertical position of matched spikes.
#
# Not like for like (read the numbers with this in mind):
# - Kilosort4's `runtime` covers all its stages (drift correction, clustering, deconvolution,
#   merging) on the device it ran on (`torch_device` in `ops.npy`); ours covers the stages
#   implemented so far.
# - Kilosort4's spikes are its final ones (after clustering and template deconvolution); ours are
#   universal-template detections. High recall is the target; precision is expected to be lower.
#
# Data (local, git-ignored): `data/kilosort4/ZFM-02370_mini.imec0.ap.short.bin` (+ `.meta`),
# `data/kilosort4/saved_results/` (see playground/README.md).

# %% [1] Settings, Recording, Probe
import json
import os
import pickletools
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
from dsp_kitchen.io import Recording
from dsp_kitchen.progress import ProgressBar
from dsp_kitchen.synapse.ml import kilosort4

DATA_DIR = Path(
    os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / "data")
)
BIN_PATH = DATA_DIR / "kilosort4" / "ZFM-02370_mini.imec0.ap.short.bin"
KS4_RESULTS = DATA_DIR / "kilosort4" / "saved_results"
OUTPUT = Path(__file__).resolve().parents[1] / "output"

#: Runtimes to time (default: the current one; e.g. `dk.runtime.available()` for all).
RUNTIMES = [dk.runtime.current()]
#: Runs per runtime: the first includes kernel compilation, the others are warm.
REPEATS = 2
#: Spike-matching tolerance (SpikeInterface default).
MATCH_MS = 0.4
#: A Kilosort4 unit counts as found when at least this fraction of its spikes is detected.
FOUND_RECALL = 0.8

rec = Recording(str(BIN_PATH))
probe = syn.ProbeLayout.from_recording(str(BIN_PATH))
if probe is None and KS4_RESULTS.exists():
    # The recording has no embedded geometry: use the probe Kilosort4 sorted with
    probe = syn.load_sorting(str(KS4_RESULTS)).probe
if probe is None:
    raise SystemExit("no probe geometry: need the .meta geometry or saved_results/")
channel_ids = probe.channel_ids()
if rec.channels != probe.total_channels:
    rec = rec.slice_samples(channels=channel_ids)
config = kilosort4.Config()
config.whitening_range = min(config.whitening_range, probe.total_channels)
fs = rec.sample_rate
duration_sec = rec.samples / fs
print(f"{rec}\n{probe}\nRuntimes: {RUNTIMES} (compiled in: {dk.runtime.available()})")


# %% [2] What Kilosort4 Recorded
def ops_summary(path):
    """Plain values of Kilosort4's `ops.npy` (a pickled dict, partly torch tensors), read by
    walking the pickle's opcodes: nothing is unpickled, so no code runs and torch is not needed."""
    raw = Path(path).read_bytes()
    header = int.from_bytes(raw[8:10], "little") if raw[6] == 1 else int.from_bytes(raw[8:12], "little")
    start = (10 if raw[6] == 1 else 12) + header
    wanted = {"runtime", "torch_device", "fs", "batch_size", "Nbatches", "nskip", "Th_universal", "n_templates", "n_pcs"}
    found, key = {}, None
    for opcode, arg, _ in pickletools.genops(raw[start:]):
        if opcode.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE"):
            if key in wanted and key not in found:
                found[key] = arg
                key = None
            else:
                key = arg
        elif key in wanted and key not in found and isinstance(arg, (int, float)) and opcode.name not in ("BINPUT", "LONG_BINPUT", "MEMOIZE"):
            found[key] = arg
            key = None
    return found


ks4_ops = ops_summary(KS4_RESULTS / "ops.npy")
ks4_times = np.load(KS4_RESULTS / "spike_times.npy").astype(np.int64).ravel()
ks4_clusters = np.load(KS4_RESULTS / "spike_clusters.npy").astype(np.int64).ravel()
ks4_positions = np.load(KS4_RESULTS / "spike_positions.npy")
# `whitening_mat_dat.npy` is the matrix Kilosort4 applied; older versions saved a Phy placeholder
# (0.005 · identity) as `whitening_mat.npy`
ks4_whitening = np.load(KS4_RESULTS / "whitening_mat_dat.npy")
ks4_channel_map = np.load(KS4_RESULTS / "channel_map.npy").astype(np.int64).ravel()
print(f"\nKilosort4 recorded: {ks4_ops}")
print(f"  {len(ks4_times):,} spikes in {len(np.unique(ks4_clusters))} units")


# %% [3] Speed
class StageTimer:
    """Records when each stage starts and ends, and draws the usual bar."""

    def __init__(self):
        self.bar = ProgressBar()
        self.stages = {}
        self._current = None

    def __call__(self, stage, step, steps, done, total, unit):
        now = time.perf_counter()
        if stage != self._current:
            self._end(now)
            self._current = stage
            self.stages[stage] = [now, now]
        self.stages[stage][1] = now
        self.bar(stage, step, steps, done, total, unit)

    def _end(self, now):
        if self._current is not None:
            self.stages[self._current][1] = now

    def close(self):
        """Ends the last stage (once: `kilosort4.run` closes its callback too)."""
        self._end(time.perf_counter())
        self._current = None
        self.bar.close()

    def durations(self):
        return {stage: end - start for stage, (start, end) in self.stages.items()}


runs = []
result = None
for runtime in RUNTIMES:
    for repeat in range(REPEATS):
        timer = StageTimer()
        t0 = time.perf_counter()
        try:
            out = kilosort4.run(rec, probe, config, progress=timer, runtime=runtime)
        finally:
            timer.close()
        wall = time.perf_counter() - t0
        runs.append({"runtime": runtime, "run": repeat + 1, "cold": repeat == 0, "wall_sec": wall, "stages_sec": timer.durations(), "spikes": len(out.spikes()["sample"])})
        result = out if result is None else result
        print(f"{runtime} run {repeat + 1} ({'cold' if repeat == 0 else 'warm'}): {wall:.1f} s, {duration_sec / wall:.1f}× real time")

print(f"\n[Speed] {duration_sec:.1f} s of recording, {rec.channels} channels at {fs:.0f} Hz")
print(f"  {'runtime':8} {'run':>3}  {'wall':>8}  {'× real time':>11}  stages")
for r in runs:
    stages = ", ".join(f"{s} {d:.1f} s" for s, d in r["stages_sec"].items())
    print(f"  {r['runtime']:8} {r['run']:>3}  {r['wall_sec']:7.1f}s  {duration_sec / r['wall_sec']:10.1f}×  {stages}")
if "runtime" in ks4_ops:
    ks4_sec = float(ks4_ops["runtime"])
    best = min(r["wall_sec"] for r in runs)
    print(
        f"  Kilosort4 (all stages, {ks4_ops.get('torch_device', '?')}): {ks4_sec:.1f} s, "
        f"{duration_sec / ks4_sec:.2f}× real time; our fastest run {best:.1f} s ({ks4_sec / best:.1f}× less, fewer stages)"
    )

# %% [4] Whitening
ours_w = np.asarray(result.whitening)
if ours_w.shape == ks4_whitening.shape and np.array_equal(np.asarray(channel_ids), ks4_channel_map):
    # Least-squares scale KS4 ≈ scale · ours (≈ µV per stored step: Kilosort4 whitens raw integers)
    scale = float(np.vdot(ours_w, ks4_whitening) / np.vdot(ours_w, ours_w))
    diff = np.linalg.norm(scale * ours_w - ks4_whitening) / np.linalg.norm(ks4_whitening)
    diag = np.corrcoef(np.diag(ours_w), np.diag(ks4_whitening))[0, 1]
    entries = np.corrcoef(ours_w.ravel(), ks4_whitening.ravel())[0, 1]
    whitening = {"scale_ks4_over_ours": scale, "relative_difference_after_scale": float(diff), "diagonal_correlation": float(diag), "entry_correlation": float(entries)}
    print(
        f"\n[Whitening] scale KS4/ours {scale:.3f}; after scaling ‖s·ours − KS4‖ / ‖KS4‖ = {diff:.3f}; "
        f"correlation: diagonal {diag:.3f}, all entries {entries:.3f}"
    )
else:
    whitening = None
    print(f"\n[Whitening] not compared: shapes {ours_w.shape} vs {ks4_whitening.shape} or a different channel order")

# %% [5] Spikes
spikes = result.spikes()
ours_t = np.asarray(spikes["sample"], dtype=np.int64)
ours_y = np.asarray(spikes["y_um"], dtype=np.float64)
order = np.argsort(ours_t)
ours_t, ours_y = ours_t[order], ours_y[order]
overall = syn.compare_spike_trains(sorted(ks4_times.tolist()), ours_t.tolist(), fs=fs, delta_time_ms=MATCH_MS)
tol = max(1, round(MATCH_MS * 1e-3 * fs))

# Nearest detection of each Kilosort4 spike (within the tolerance)
if len(ours_t) == 0:
    matched, dy = np.zeros(len(ks4_times), dtype=bool), np.empty(0)
else:
    last = len(ours_t) - 1
    idx = np.clip(np.searchsorted(ours_t, ks4_times), 1, max(last, 1))
    left, right = ours_t[idx - 1], ours_t[np.minimum(idx, last)]
    nearest = np.where(np.abs(ks4_times - left) <= np.abs(ks4_times - right), idx - 1, np.minimum(idx, last))
    matched = np.abs(ours_t[nearest] - ks4_times) <= tol
    dy = np.abs(ours_y[nearest[matched]] - ks4_positions[matched, 1])

units = np.unique(ks4_clusters)
unit_recall = np.array([matched[ks4_clusters == u].mean() for u in units])
found = int((unit_recall >= FOUND_RECALL).sum())
print(
    f"\n[Spikes] Kilosort4 {len(ks4_times):,} vs ours {len(ours_t):,} (±{MATCH_MS} ms): "
    f"recall {overall['recall']:.3f}, precision {overall['precision']:.3f}"
)
print(f"  Units: {found}/{len(units)} with recall ≥ {FOUND_RECALL:.0%}; median unit recall {np.median(unit_recall):.3f}")
if len(dy):
    print(f"  Matched spikes: vertical position |Δy| median {np.median(dy):.1f} µm, 90th percentile {np.percentile(dy, 90):.1f} µm")

# %% [6] Save
OUTPUT.mkdir(exist_ok=True)
summary = {
    "recording": {"file": BIN_PATH.name, "channels": rec.channels, "sample_rate_hz": fs, "duration_sec": duration_sec},
    "runs": runs,
    "kilosort4_recorded": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in ks4_ops.items()},
    "whitening": whitening,
    "spikes": {
        "kilosort4": int(len(ks4_times)),
        "ours": int(len(ours_t)),
        "recall": overall["recall"],
        "precision": overall["precision"],
        "units_found": found,
        "units": int(len(units)),
        "median_unit_recall": float(np.median(unit_recall)),
        "median_abs_dy_um": float(np.median(dy)) if len(dy) else None,
    },
}
path = OUTPUT / "kilosort4_benchmark.json"
path.write_text(json.dumps(summary, indent=2))
print(f"\nSaved {path}")

# %% [7] Plot
if HAS_PLT:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    labels = [f"{r['runtime']} #{r['run']}" for r in runs]
    bottom = np.zeros(len(runs))
    for stage in dict.fromkeys(s for r in runs for s in r["stages_sec"]):
        values = np.array([r["stages_sec"].get(stage, 0.0) for r in runs])
        axes[0].bar(labels, values, bottom=bottom, label=stage)
        bottom += values
    axes[0].set_ylabel("Wall time (s)")
    axes[0].set_title("Our stages per run", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[1].hist(unit_recall, bins=20, range=(0, 1))
    axes[1].axvline(FOUND_RECALL, color="k", linestyle="--")
    axes[1].set_xlabel("Fraction of the unit's spikes detected")
    axes[1].set_ylabel("Kilosort4 units")
    axes[1].set_title("Recall per Kilosort4 unit", fontweight="bold")
    axes[2].hist(dy, bins=40, range=(0, max(1.0, np.percentile(dy, 99)) if len(dy) else 1.0))
    axes[2].set_xlabel("|Δy| (µm)")
    axes[2].set_title("Vertical position of matched spikes", fontweight="bold")
    plt.tight_layout()
    plt.show()

# %%
