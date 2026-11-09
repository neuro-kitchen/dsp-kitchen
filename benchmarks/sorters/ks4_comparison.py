"""A sorter's output against Kilosort4's saved results on the Neuropixels test recording: spikes
per channel (where and when) and units (each Kilosort4 unit against the unit of ours holding most
of its spikes, matched in time and place). Used by the `*_validation.py` scripts.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from ks4_reference import BIN_PATH, KS4_RESULTS, Reference, compare
from sorter_cli import RESULTS, flags_from_argv, flatten, load, save_json, sort

#: Spikes match within this time (samples at 30 kHz).
MATCH_SAMPLES = 12


def probe_file() -> Path:
    """The geometry Kilosort4 sorted with, as a Kilosort probe file (the recording carries none)."""
    channel_map = np.load(KS4_RESULTS / "channel_map.npy").astype(int).ravel()
    positions = np.load(KS4_RESULTS / "channel_positions.npy").astype(float)
    path = RESULTS / "ks4_probe.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"chanMap": channel_map.tolist(), "xc": positions[:, 0].tolist(), "yc": positions[:, 1].tolist(), "kcoords": [0] * len(channel_map), "n_chan": 385}))
    return path


def validate(sorter: str, near_um: float) -> tuple[dict, Path]:
    """Sorts (once; `--rerun` sorts again; `name=value` sets a setting), compares, writes
    `results/<sorter>_validation.json`. Returns the record and the sorting's path."""
    output = RESULTS / f"{sorter}_validation" / "ours.sorting.zarr"
    run = None
    if "--rerun" in sys.argv or not output.exists():
        run = sort(sorter, BIN_PATH, output, flags=flags_from_argv(sys.argv[1:]), probe_file=probe_file())
        print(f"{sorter}: {run.units} units, {run.spikes:,} spikes in {run.wall_sec:.1f} s")
    ours = load(output)
    ref = Reference.load()
    times, units, locs = flatten(ours)
    located = np.isfinite(locs).all(axis=1)
    if not located.all():
        print(f"  {(~located).sum()} spikes have no location: left out of the comparison")
        times, units, locs = times[located], units[located], locs[located]

    print("\n[Spikes per channel]")
    stats = compare(ref, times, ref.nearest_channel(locs), label="ours", table_rows=10)

    ks_units = np.load(KS4_RESULTS / "spike_clusters.npy").ravel()
    ks_times = np.load(KS4_RESULTS / "spike_times.npy").astype(np.int64).ravel()
    ks_pos = np.load(KS4_RESULTS / "spike_positions.npy")[:, :2]
    with open(KS4_RESULTS / "cluster_KSLabel.tsv") as f:
        labels = {int(r["cluster_id"]): r["KSLabel"] for r in csv.DictReader(f, delimiter="\t")}
    shift = stats["offset_peak"] or 0
    start, end = np.searchsorted(times, ks_times + shift - MATCH_SAMPLES), np.searchsorted(times, ks_times + shift + MATCH_SAMPLES, side="right")
    match = np.full(len(ks_times), -1)
    for i in np.flatnonzero(end > start):
        c = np.arange(start[i], end[i])
        c = c[np.hypot(*(locs[c] - ks_pos[i]).T) <= near_um]
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
    table = ours.units()
    our_good = sum(label in ("good", "SingleUnit") for label in table["label"])

    def summary(sel) -> dict:
        r = rows[sel]
        q = lambda a: {"median": float(np.median(a)), "p25": float(np.percentile(a, 25))}  # noqa: E731
        return {"units": int(sel.sum()), "found": q(r[:, 1]), "not_split": q(r[:, 2]), "precision": q(r[:, 3]), "accuracy": q(r[:, 4]),
                "accuracy_ge_0_8": float(np.mean(r[:, 4] >= 0.8)), "accuracy_ge_0_5": float(np.mean(r[:, 4] >= 0.5))}

    record = {
        "sorter": sorter,
        "near_um": near_um,
        "wall_sec": run.wall_sec if run else None,
        "exit_code": run.exit_code if run else None,
        "stages_sec": run.stages_sec if run else None,
        "spikes": int(len(times)),
        "units": int(len(ours_n)),
        "units_good": int(our_good),
        "ks4_spikes": int(len(ks_times)),
        "ks4_units": int(len(rows)),
        "ks4_units_good": int(good.sum()),
        "ks4_spikes_found": float(np.mean(match >= 0)),
        "offset_peak_samples": stats["offset_peak"],
        "per_channel": stats,
        "all_units": summary(np.ones(len(rows), bool)),
        "good_units": summary(good),
    }
    print(f"\n[Units] ours {len(ours_n)} ({our_good} good) vs Kilosort4 {len(rows)} ({good.sum()} good); Kilosort4 spikes found in time and place: {np.mean(match >= 0):.3f}")
    print(json.dumps({"good": record["good_units"]}, indent=1))
    save_json(f"{sorter}_validation", record)
    return record, output
