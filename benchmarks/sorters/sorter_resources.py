# %% [markdown]
# # Time and memory of one sorter run
# Runs Kilosort4 (Neuropixels test recording) or EMUsort (HD-EMG recording) from the raw file to
# sorted units, `--repeats` times, and records wall time and time per stage, peak host RSS of the
# sorter's process, peak GPU memory (`nvidia-smi`, minus the idle reading; NVIDIA only) and whether
# every run gives the same spikes and units.
#
#     uv run python benchmarks/sorters/sorter_resources.py emusort                 # whole recording
#     uv run python benchmarks/sorters/sorter_resources.py emusort --seconds 200
#     uv run python benchmarks/sorters/sorter_resources.py ks4
#
# Writes `results/sorter_resources_<sorter>.json`; `sorter_resources_table.py` draws them. Needs a
# release build of `dspk-cli-sorter` to mean anything (see `sorter_cli.py`).
#
# Data (local, git-ignored): `.datasets/kilosort4/ZFM-02370_mini.imec0.ap.short.bin` (+ `saved_results/`
# for the probe), `.datasets/nwb/15-25-33_meps.nwb.zarr` (HDEMG series).

# %%
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

import dspk
from ks4_comparison import probe_file
from ks4_reference import BIN_PATH, DATA_DIR
from sorter_cli import RESULTS, gpu_name, identical, load, save_json, sort

NWB_PATH = DATA_DIR / "nwb" / "15-25-33_meps.nwb.zarr"
ELECTRODE_PITCH_UM = 100.0
EMG_CHANNELS = 32

parser = argparse.ArgumentParser()
parser.add_argument("sorter", choices=["ks4", "emusort"])
parser.add_argument("--seconds", type=float, default=None, help="sort the first N s (default: all)")
parser.add_argument("--repeats", type=int, default=2, help="runs; the first may compile and tune")
args = parser.parse_args()

if args.sorter == "ks4":
    name, recording, source = "kilosort4", BIN_PATH, "Neuropixels 1.0"
    options = dict(probe_file=probe_file())
else:
    name, recording, source = "emusort", NWB_PATH, "HD-EMG 4 × 8 grid"
    options = dict(source="/acquisition/HDEMG", probe="hdemg4x8", pitch_um=ELECTRODE_PITCH_UM)
if args.seconds is not None:
    options["duration_s"] = args.seconds

# %% Runs
OUTPUT = RESULTS / f"sorter_resources_{args.sorter}"
runs = []
for repeat in range(args.repeats):
    run = sort(name, recording, OUTPUT / f"run{repeat}.sorting.zarr", **options)
    runs.append(asdict(run))
    print(f"run {repeat + 1}: {run.wall_sec:.1f} s, RSS {run.peak_rss_mib / 1024:.2f} GiB, GPU {run.peak_gpu_mib}, {run.spikes:,} spikes, {run.units} units", flush=True)
first = load(OUTPUT / "run0.sorting.zarr")
reproducible = all(identical(first, load(OUTPUT / f"run{i}.sorting.zarr")) for i in range(1, args.repeats))
print(f"reproducible: {reproducible}")
fs, duration = first.sample_rate, first.total_samples / first.sample_rate
channels = len(first.probe.channels) if args.sorter == "ks4" else EMG_CHANNELS

# %% Save
save_json(f"sorter_resources_{args.sorter}", {"sorter": args.sorter, "source": source, "sample_rate": fs, "duration_sec": duration, "runtime": dspk.runtime.current(), "channels": channels, "signal_f32_gib": channels * first.total_samples * 4 / 2**30, "gpu": gpu_name(), "process_max_rss_mib": max(r["peak_rss_mib"] for r in runs), "runs": runs, "reproducible": reproducible})
