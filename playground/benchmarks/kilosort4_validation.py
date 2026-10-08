# %% [markdown]
# # Kilosort4 validation: our output vs Kilosort4's saved results
# Checks that our Kilosort4 produces output comparable with what Kilosort4 recorded in
# `data/kilosort4/saved_results/` (a Phy folder), without running Kilosort4 itself:
# 1. **Export**: our sorting is written as a Phy folder and a `.sorting.zarr` and read back; every
#    comparison below uses the read-back copy; the Phy files are checked against Kilosort4's.
# 2. **Spikes per channel**: unit ids differ between sorters, so spikes are compared by *where*
#    (the contact nearest each spike's own position, for both sorters) and *when* (±`MATCH_MS`).
#    The time-difference histogram must peak at 0 (both report waveform troughs).
# 3. **Units**: each Kilosort4 unit against the unit of ours holding most of its spikes, matched
#    in time **and** place (time alone matches by chance at this spike density).
#
# Data (local, git-ignored): `data/kilosort4/ZFM-02370_mini.imec0.ap.short.bin` (+ `.meta`),
# `data/kilosort4/saved_results/` (see playground/README.md). Results: the book's *Benchmarks*.

# %% [1] Settings
import csv
import sys
from pathlib import Path

import numpy as np

import dsp_kitchen.synapse as syn
from dsp_kitchen.io import Recording
from dsp_kitchen.progress import ProgressBar
from dsp_kitchen.synapse.ml import kilosort4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ks4_reference import BIN_PATH, KS4_RESULTS, Reference, compare, read_params  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output" / "kilosort4_validation"
PHY_OUT = OUTPUT / "phy"
ZARR_OUT = OUTPUT / "ours.sorting.zarr"
#: Spikes match within this time (samples at 30 kHz: 12) and distance.
MATCH_SAMPLES, NEAR_UM = 12, 40.0
#: Reuse the exported sorting of an earlier run (pass `--rerun` to sort again).
RERUN = "--rerun" in sys.argv

ref = Reference.load()  # Kilosort4's spikes placed by their own positions
fs = ref.fs

# %% [2] Our Sorting (Run Once, Exported, Read Back)
if RERUN or not ZARR_OUT.exists():
    rec = Recording(str(BIN_PATH))
    probe = syn.ProbeLayout.from_recording(str(BIN_PATH)) or syn.load_sorting(str(KS4_RESULTS)).probe
    if rec.channels != probe.total_channels:
        rec = rec.slice_samples(channels=probe.channel_ids())
    config = kilosort4.Config()
    config.whitening_range = min(config.whitening_range, probe.total_channels)
    result = kilosort4.run(rec, probe, config, progress=ProgressBar())
    print(f"{result} | {result.n_units} units | reproducible={result.reproducible} | device: {result.device}")
    sorting = result.to_sorting_output(probe)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    syn.save_sorting(sorting, str(PHY_OUT), format="phy")
    syn.save_sorting(sorting, str(ZARR_OUT), format="sorting-zarr")

ours_zarr, ours_phy = syn.load_sorting(str(ZARR_OUT)), syn.load_sorting(str(PHY_OUT))


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


times, units, locs = flatten(ours_zarr)

# %% [3] Export Round Trip
print("\n[Export]")
phy_times, phy_units, _ = flatten(ours_phy)
ours_files = {p.name for p in PHY_OUT.iterdir()}
theirs_files = {p.name for p in KS4_RESULTS.iterdir() if p.suffix in (".npy", ".tsv", ".py")}
# Kilosort-internal files Phy does not need (Phy assumes identity whitening when absent)
optional = {"ops.npy", "kept_spikes.npy", "templates_ind.npy", "whitening_mat_dat.npy", "whitening_mat.npy", "whitening_mat_inv.npy", "cluster_Amplitude.tsv", "cluster_ContamPct.tsv"}
missing = sorted(theirs_files - ours_files - optional)
for name, ok in [
    ("zarr and phy read back the same spike times", np.array_equal(times, phy_times)),
    ("zarr and phy read back the same units", np.array_equal(units, phy_units)),
    ("phy sample rate", float(read_params(PHY_OUT)["sample_rate"]) == fs),
    (f"Phy files Kilosort4 writes (missing: {missing or 'none'})", not missing),
]:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"  ours: {ours_zarr.num_units} units, {ours_zarr.total_spikes:,} spikes")

# %% [4] Spikes per Channel
print("\n[Spikes per channel]")
stats = compare(ref, times, ref.nearest_channel(locs), label="ours", table_rows=10)
print(f"  time offset peak {stats['offset_peak']:+d} samples → {'PASS' if stats['offset_peak'] in (-1, 0, 1) else 'FAIL'}")

# %% [5] Units
ks_units = np.load(KS4_RESULTS / "spike_clusters.npy").ravel()
ks_times = np.load(KS4_RESULTS / "spike_times.npy").astype(np.int64).ravel()
ks_pos = np.load(KS4_RESULTS / "spike_positions.npy")[:, :2]
labels = {int(r["cluster_id"]): r["KSLabel"] for r in csv.DictReader(open(KS4_RESULTS / "cluster_KSLabel.tsv"), delimiter="\t")}
# Each Kilosort4 spike's match: our spike within ±MATCH_SAMPLES and NEAR_UM, nearest in time
start, end = np.searchsorted(times, ks_times - MATCH_SAMPLES), np.searchsorted(times, ks_times + MATCH_SAMPLES, side="right")
match = np.full(len(ks_times), -1)
for i in np.flatnonzero(end > start):
    c = np.arange(start[i], end[i])
    c = c[np.hypot(*(locs[c] - ks_pos[i]).T) <= NEAR_UM]
    if len(c):
        match[i] = c[np.argmin(np.abs(times[c] - ks_times[i]))]
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
for name, sel in (("all", np.ones(len(rows), bool)), ("good", good)):
    r = rows[sel]

    def f(a):
        return f"{np.median(a):.2f} ({np.percentile(a, 25):.2f})"

    print(f"  {name:4} {sel.sum():3} | found {f(r[:, 1])} | not split {f(r[:, 2])} | precision {f(r[:, 3])} | accuracy {f(r[:, 4])}; ≥0.8: {np.mean(r[:, 4] >= 0.8):.2f}, ≥0.5: {np.mean(r[:, 4] >= 0.5):.2f}")

# %%
