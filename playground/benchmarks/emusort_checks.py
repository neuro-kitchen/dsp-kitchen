# %% [markdown]
# # EMUsort checks on HD-EMG (no ground truth)
# Runs EMUsort on a segment of the HD-EMG recording and checks what can be checked without a
# reference sorting (the same checks that located problems against Kilosort4's saved results):
# 1. **Run**: time per stage, spikes, units, channel delays.
# 2. **Reproducibility**: a second run gives the same spikes and units.
# 3. **Units**: spikes per unit, refractory-period violations (motor units fire at most every
#    ~10 ms), presence over the segment, firing rate, template norm.
# 4. **Templates**: how many learned templates the units yield, and the most similar pairs left.
#
#     uv run python playground/benchmarks/emusort_checks.py [segment seconds, default 200] [name=value ...]
#
# `name=value` overrides a setting (e.g. `do_notch=False global_merges=False`).
#
# Data (local, git-ignored): `data/nwb/15-25-33_meps.nwb.zarr` (HDEMG series).

# %% [1] Settings
import ast
import os
import sys
import time
from pathlib import Path

import numpy as np

import dsp_kitchen.synapse as syn
from dsp_kitchen.io import Recording, list_sources
from dsp_kitchen.synapse.ml import emusort

DATA_DIR = Path(os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / "data"))
NWB_PATH = DATA_DIR / "nwb" / "15-25-33_meps.nwb.zarr"
GRID_ROWS, GRID_COLS, ELECTRODE_PITCH_UM = 4, 8, 100.0
#: Segment sorted (s): long enough for stable units, short enough to iterate on.
POSITIONAL = [a for a in sys.argv[1:] if "=" not in a]
OVERRIDES = [a for a in sys.argv[1:] if "=" in a]
SEGMENT_SEC = float(POSITIONAL[0]) if POSITIONAL else 200.0
#: Spikes closer than this in one unit violate a motor unit's refractory period.
REFRACTORY_MS = 2.0
#: Presence: share of 10 s bins where the unit fires.
PRESENCE_BIN_SEC = 10.0

hdemg = next((s["id"] for s in list_sources(str(NWB_PATH)) if "HDEMG" in s["id"]), None)
rec = Recording(str(NWB_PATH), source=hdemg).slice_time(start_sec=0.0, end_sec=SEGMENT_SEC)
probe = syn.ProbeLayout.hdemg_grid("HD-EMG 4x8", GRID_ROWS, GRID_COLS, ELECTRODE_PITCH_UM)
config = emusort.Config()
for arg in OVERRIDES:
    name, value = arg.split("=", 1)
    *path, field = name.split(".")
    target = config
    for part in path:
        target = getattr(target, part)
    setattr(target, field, ast.literal_eval(value))
    print(f"setting {name} = {getattr(target, field)!r}")
fs = rec.sample_rate
print(f"{rec}\n{config}")


# %% [2] Run (twice: reproducibility)
def run():
    stages = {}

    def progress(stage, step, steps, done, total, unit):
        stages.setdefault(stage, [time.perf_counter(), 0.0])[1] = time.perf_counter()

    t0 = time.perf_counter()
    result = emusort.run(rec, probe, config, progress=progress)
    return result, time.perf_counter() - t0, {k: e - s for k, (s, e) in stages.items()}


result, wall, stages = run()
again, wall_warm, stages_warm = run()
spikes, spikes_again = result.spikes(), again.spikes()
delays, reference = result.channel_delays
print(f"\n[Run] {SEGMENT_SEC:.0f} s of {rec.channels} channels in {wall:.1f} s ({SEGMENT_SEC / wall:.1f}× real time)")
print("  cold: " + ", ".join(f"{k} {v:.2f} s" for k, v in stages.items()))
print(f"[Run, warm] {wall_warm:.1f} s ({SEGMENT_SEC / wall_warm:.1f}× real time)")
print("  warm: " + ", ".join(f"{k} {v:.2f} s" for k, v in stages_warm.items()))
print(f"  channel delays vs channel {reference}: {delays}")
print(f"  {len(spikes['sample']):,} spikes, {result.n_units} units, {result.n_learned_templates} learned templates")
same = np.array_equal(spikes["sample"], spikes_again["sample"]) and np.array_equal(spikes["unit"], spikes_again["unit"])
print(f"\n[Reproducibility] second run identical: {same}")

# %% [3] Units
t = np.asarray(spikes["sample"], dtype=np.int64)
u = np.asarray(spikes["unit"])
amp = np.asarray(spikes["amplitude"])
rows = []
for unit in np.unique(u):
    ts = np.sort(t[u == unit])
    isi_ms = np.diff(ts) / fs * 1e3
    bins = np.bincount((ts / fs // PRESENCE_BIN_SEC).astype(int), minlength=int(np.ceil(SEGMENT_SEC / PRESENCE_BIN_SEC)))
    rows.append((unit, len(ts), len(ts) / SEGMENT_SEC, np.mean(isi_ms < REFRACTORY_MS) if len(isi_ms) else 0.0, np.mean(bins > 0), np.median(amp[u == unit])))
rows = np.array(rows)
n, rate, viol, presence = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
print(f"\n[Units] {len(rows)} units: spikes per unit median {np.median(n):.0f} (min {n.min():.0f}, max {n.max():.0f})")
print(f"  firing rate median {np.median(rate):.2f} Hz; presence median {np.median(presence):.2f}")
print(f"  refractory violations (ISI < {REFRACTORY_MS} ms): median {np.median(viol):.3f}; units above 1%: {(viol > 0.01).sum()}, above 5%: {(viol > 0.05).sum()}")
print(f"  {'unit':>4} {'spikes':>7} {'Hz':>6} {'ISI<2ms':>8} {'present':>8} {'amp':>6}")
for r in rows[np.argsort(-rows[:, 1])][:12]:
    print(f"  {int(r[0]):>4} {int(r[1]):>7} {r[2]:>6.2f} {r[3]:>8.3f} {r[4]:>8.2f} {r[5]:>6.1f}")

# %% [4] Templates and export
sorting = result.to_sorting_output(probe)
print(f"\n[Export] {sorting}")
with_template = sum(1 for uid in sorting.unit_ids() if sorting.unit_metrics(uid)["primary_channel"] is not None)
print(f"  units with a primary channel (template present): {with_template}/{sorting.num_units}")
