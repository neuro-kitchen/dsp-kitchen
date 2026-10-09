# %% [markdown]
# # MountainSort 5 validation: our output vs Kilosort4's saved results
# There is no MountainSort 5 reference on this recording: its units are compared with Kilosort4's
# (`data/kilosort4/saved_results/`), as a second, independent sorter on the same data.
# 1. **Spikes per channel**: spikes compared by *where* (the contact nearest each spike: ours is the
#    detection channel) and *when* (±`MATCH_MS`); the time-difference histogram should peak near 0
#    (both report the waveform's peak).
# 2. **Units**: each Kilosort4 unit against the unit of ours holding most of its spikes, matched in
#    time **and** place.
#
# `--set name=value` changes a setting (e.g. `--set scheme=1`); `--rerun` sorts again instead of
# reading the exported sorting of an earlier run.
#
# Data (local, git-ignored): `data/kilosort4/ZFM-02370_mini.imec0.ap.short.bin` (+ `.meta`),
# `data/kilosort4/saved_results/` (see playground/README.md). Results: the book's *Benchmarks*.

# %% [1] Settings
import ast
import csv
import sys
import time
from pathlib import Path

import numpy as np

import dsp_kitchen.synapse as syn
from dsp_kitchen.io import Recording
from dsp_kitchen.progress import ProgressBar
from dsp_kitchen.synapse.ml import mountainsort5

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ks4_reference import BIN_PATH, KS4_RESULTS, Reference, compare  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output" / "mountainsort5_validation"
ZARR_OUT = OUTPUT / "ours.sorting.zarr"
#: Spikes match within this time (samples at 30 kHz: 12) and distance (ours: the detection
#: channel's position, Kilosort4: its estimated position).
MATCH_SAMPLES, NEAR_UM = 12, 60.0
RERUN = "--rerun" in sys.argv

ref = Reference.load()
fs = ref.fs

# %% [2] Our Sorting (Run Once, Exported, Read Back)
if RERUN or not ZARR_OUT.exists():
    rec = Recording(str(BIN_PATH))
    probe = syn.ProbeLayout.from_recording(str(BIN_PATH)) or syn.load_sorting(str(KS4_RESULTS)).probe
    if rec.channels != probe.total_channels:
        rec = rec.slice_samples(channels=probe.channel_ids())
    config = mountainsort5.Config()
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith("--"):
            name, value = arg.split("=", 1)
            setattr(config, name, ast.literal_eval(value))
            print(f"setting {name} = {getattr(config, name)!r}")
    start = time.perf_counter()
    result = mountainsort5.run(rec, probe, config, progress=ProgressBar())
    print(f"{result} in {time.perf_counter() - start:.1f} s")
    sorting = result.to_sorting_output(probe)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    syn.save_sorting(sorting, str(ZARR_OUT), format="sorting-zarr")

ours = syn.load_sorting(str(ZARR_OUT))


def flatten(sorting):
    """All spikes of a sorting as (samples, unit ids, locations [n, 2] µm), in time order."""
    times, units, locs = [], [], []
    for u in sorting.unit_ids():
        t = np.asarray(sorting.spike_train(u), dtype=np.int64)
        times.append(t)
        units.append(np.full(len(t), u))
        locs.append(np.asarray(sorting.spike_locations(u), dtype=np.float64)[:, :2])
    times, units, locs = np.concatenate(times), np.concatenate(units), np.concatenate(locs)
    order = np.argsort(times, kind="stable")
    return times[order], units[order], locs[order]


times, units, locs = flatten(ours)
print(f"ours: {ours.num_units} units, {ours.total_spikes:,} spikes")

# %% [3] Spikes per Channel
print("\n[Spikes per channel]")
stats = compare(ref, times, ref.nearest_channel(locs), label="ours", table_rows=10)
print(f"  time offset peak {stats['offset_peak']:+d} samples")

# %% [4] Units
ks_units = np.load(KS4_RESULTS / "spike_clusters.npy").ravel()
ks_times = np.load(KS4_RESULTS / "spike_times.npy").astype(np.int64).ravel()
ks_pos = np.load(KS4_RESULTS / "spike_positions.npy")[:, :2]
labels = {int(r["cluster_id"]): r["KSLabel"] for r in csv.DictReader(open(KS4_RESULTS / "cluster_KSLabel.tsv"), delimiter="\t")}
shift = stats["offset_peak"]
start, end = np.searchsorted(times, ks_times + shift - MATCH_SAMPLES), np.searchsorted(times, ks_times + shift + MATCH_SAMPLES, side="right")
match = np.full(len(ks_times), -1)
for i in np.flatnonzero(end > start):
    c = np.arange(start[i], end[i])
    c = c[np.hypot(*(locs[c] - ks_pos[i]).T) <= NEAR_UM]
    if len(c):
        match[i] = c[np.argmin(np.abs(times[c] - shift - ks_times[i]))]
ours_n = dict(zip(*np.unique(units, return_counts=True)))
rows = []
for k in np.unique(ks_units):
    sel = ks_units == k
    m = match[sel]
    m = m[m >= 0]
    if len(m) == 0:
        rows.append((k, 0.0, 0.0, 0.0, 0.0))
        continue
    cu, cc = np.unique(units[m], return_counts=True)
    b = int(np.argmax(cc))
    hits, n = cc[b], sel.sum()
    rows.append((k, len(m) / n, hits / len(m), hits / ours_n[cu[b]], hits / (n + ours_n[cu[b]] - hits)))
rows = np.array(rows)
good = np.array([labels.get(int(k)) == "good" for k in rows[:, 0]])
print(f"\n[Units] ours {len(ours_n)} vs Kilosort4 {len(rows)} ({good.sum()} good); Kilosort4 spikes found in time and place: {np.mean(match >= 0):.3f}")
our_good = sum(ours.unit_metrics(u)["quality_label"] in ("SingleUnit", "good") for u in ours.unit_ids())
print(f"  our labels: {our_good} good of {len(ours_n)} (Kilosort4: {good.sum()} of {len(rows)})")
for name, sel in (("all", np.ones(len(rows), bool)), ("good", good)):
    r = rows[sel]

    def f(a):
        return f"{np.median(a):.2f} ({np.percentile(a, 25):.2f})"

    print(f"  {name:4} {sel.sum():3} | found {f(r[:, 1])} | not split {f(r[:, 2])} | precision {f(r[:, 3])} | accuracy {f(r[:, 4])}; ≥0.8: {np.mean(r[:, 4] >= 0.8):.2f}, ≥0.5: {np.mean(r[:, 4] >= 0.5):.2f}")
