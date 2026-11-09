# %% [markdown]
# # EMUsort checks on HD-EMG (no ground truth)
# Runs EMUsort on a segment of the HD-EMG recording, twice, and checks what can be checked without
# a reference sorting: reproducibility, spikes per unit, refractory-period violations (motor units
# fire at most every ~10 ms) and presence over the segment.
#
#     uv run python benchmarks/sorters/emusort_checks.py [segment seconds, default 200] [name=value ...]
#
# Writes `results/emusort_checks.json`. Needs `dspk-cli-sorter` (see `sorter_cli.py`).
#
# Data (local, git-ignored): `.datasets/nwb/15-25-33_meps.nwb.zarr` (HDEMG series).

# %%
import os
import sys
from pathlib import Path

import numpy as np

from sorter_cli import RESULTS, flags_from_argv, flatten, identical, load, save_json, sort

DATA_DIR = Path(os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / ".datasets"))
NWB_PATH = DATA_DIR / "nwb" / "15-25-33_meps.nwb.zarr"
ELECTRODE_PITCH_UM = 100.0
POSITIONAL = [a for a in sys.argv[1:] if "=" not in a]
SEGMENT_SEC = float(POSITIONAL[0]) if POSITIONAL else 200.0
#: Spikes closer than this in one unit violate a motor unit's refractory period.
REFRACTORY_MS = 2.0
#: Presence: share of 10 s bins where the unit fires.
PRESENCE_BIN_SEC = 10.0

OUTPUT = RESULTS / "emusort_checks"
options = dict(flags=flags_from_argv(sys.argv[1:]), source="/acquisition/HDEMG", probe="hdemg4x8", pitch_um=ELECTRODE_PITCH_UM, duration_s=SEGMENT_SEC)

# %% Twice: reproducibility
first = sort("emusort", NWB_PATH, OUTPUT / "first.sorting.zarr", **options)
second = sort("emusort", NWB_PATH, OUTPUT / "second.sorting.zarr", **options)
sorting = load(OUTPUT / "first.sorting.zarr")
same = identical(sorting, load(OUTPUT / "second.sorting.zarr"))
fs = sorting.sample_rate
print(f"\n[Run] {SEGMENT_SEC:.0f} s in {first.wall_sec:.1f} s ({SEGMENT_SEC / first.wall_sec:.1f}× real time); second run {second.wall_sec:.1f} s")
print(f"[Reproducibility] second run identical: {same}")

# %% Units
t, u, _ = flatten(sorting)
rows = []
for unit in np.unique(u):
    ts = t[u == unit]
    isi_ms = np.diff(ts) / fs * 1e3
    bins = np.bincount((ts / fs // PRESENCE_BIN_SEC).astype(int), minlength=int(np.ceil(SEGMENT_SEC / PRESENCE_BIN_SEC)))
    rows.append((len(ts), np.mean(isi_ms < REFRACTORY_MS) if len(isi_ms) else 0.0, np.mean(bins > 0)))
rows = np.array(rows)
print(f"[Units] {len(rows)} units; spikes per unit median {np.median(rows[:, 0]):.0f}; above 1% / 5% violations: {(rows[:, 1] > 0.01).sum()} / {(rows[:, 1] > 0.05).sum()}")

save_json("emusort_checks", {
    "segment_sec": SEGMENT_SEC,
    "wall_sec": [first.wall_sec, second.wall_sec],
    "stages_sec": first.stages_sec,
    "second_stages_sec": second.stages_sec,
    "spikes": first.spikes,
    "units": first.units,
    "reproducible": same,
    "units_over_1pct_violations": int((rows[:, 1] > 0.01).sum()),
    "units_over_5pct_violations": int((rows[:, 1] > 0.05).sum()),
    "median_presence": float(np.median(rows[:, 2])),
})
