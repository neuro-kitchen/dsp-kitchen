# %% [markdown]
# # Kilosort4 speed per stage and runtime, and its spikes against Kilosort4's own
# Runs our Kilosort4 over the Neuropixels test recording `--repeats` times per runtime: wall time and
# time per stage, as a multiple of real time. The first run on a device compiles and tunes unless
# the kernel cache has them already. Spikes are compared in time only, per Kilosort4 unit; for time
# *and* place see `kilosort4_validation.py` (time alone matches by chance at this spike density).
#
#     uv run python benchmarks/sorters/kilosort4_benchmark.py [--runtimes cuda wgpu cpu] [--repeats 2]
#
# Writes `results/kilosort4_benchmark.json`. Needs a release build of `dspk-cli-sorter` to mean
# anything (see `sorter_cli.py`).

# %%
import argparse

import numpy as np

import dspk
from ks4_comparison import probe_file
from ks4_reference import BIN_PATH, KS4_RESULTS, read_params
from sorter_cli import RESULTS, flatten, load, save_json, sort

parser = argparse.ArgumentParser()
parser.add_argument("--runtimes", nargs="+", default=None)
parser.add_argument("--repeats", type=int, default=2)
args = parser.parse_args()
RUNTIMES = args.runtimes or [dspk.runtime.current()]
#: Spike-matching tolerance (SpikeInterface default).
MATCH_MS = 0.4
#: A Kilosort4 unit counts as found when at least this fraction of its spikes is detected.
FOUND_RECALL = 0.8

fs = float(read_params(KS4_RESULTS)["sample_rate"])
ks_times = np.load(KS4_RESULTS / "spike_times.npy").astype(np.int64).ravel()
ks_units = np.load(KS4_RESULTS / "spike_clusters.npy").astype(np.int64).ravel()
output = RESULTS / "kilosort4_benchmark" / "ours.sorting.zarr"

# %% Speed
runs = []
for runtime in RUNTIMES:
    for repeat in range(args.repeats):
        run = sort("kilosort4", BIN_PATH, output, probe_file=probe_file(), runtime=runtime)
        duration = load(output).total_samples / fs
        runs.append({"runtime": runtime, "run": repeat + 1, "wall_sec": run.wall_sec, "stages_sec": run.stages_sec, "real_time": duration / run.wall_sec, "units": run.units, "spikes": run.spikes})
        print(f"{runtime} run {repeat + 1}: {run.wall_sec:.1f} s, {duration / run.wall_sec:.1f}× real time")

# %% Spikes, in time only
ours, _, _ = flatten(load(output))
compared = dspk.synapse.metrics.compare_trains(np.sort(ks_times), ours, tolerance_samples=max(1, round(MATCH_MS * 1e-3 * fs)))
idx = np.clip(np.searchsorted(ours, ks_times), 1, len(ours) - 1)
nearest = np.minimum(np.abs(ks_times - ours[idx - 1]), np.abs(ks_times - ours[idx]))
matched = nearest <= max(1, round(MATCH_MS * 1e-3 * fs))
units = np.unique(ks_units)
recall = np.array([matched[ks_units == u].mean() for u in units])
print(f"[Spikes] recall {compared['recall']:.3f}, precision {compared['precision']:.3f}; units with recall ≥ {FOUND_RECALL:.0%}: {(recall >= FOUND_RECALL).sum()}/{len(units)}")
save_json("kilosort4_benchmark", {"runs": runs, "recall": compared["recall"], "precision": compared["precision"], "units_found": int((recall >= FOUND_RECALL).sum()), "ks4_units": int(len(units)), "median_unit_recall": float(np.median(recall))})
