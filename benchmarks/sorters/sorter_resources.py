# %% [markdown]
# # dsp-kitchen benchmark: time and memory of one sorter run
# Runs Kilosort4 (Neuropixels test recording) or EMUsort (HD-EMG recording) from the raw file to
# sorted units, `--repeats` times in one process, and records:
# 1. **Time**: wall time and time per stage (a stage lasts from its first progress report to the
#    next stage's); the first run includes kernel compilation (cold), later runs are warm.
# 2. **Memory**: peak host RSS of the process and peak GPU memory in use (`nvidia-smi`, minus the
#    idle reading before the run; NVIDIA only, left out without `nvidia-smi`), sampled every 50 ms.
# 3. **Reproducibility**: whether every run gives the same spikes and units.
#
#     uv run python benchmarks/sorters/sorter_resources.py emusort                 # whole recording
#     uv run python benchmarks/sorters/sorter_resources.py emusort --seconds 200
#     uv run python benchmarks/sorters/sorter_resources.py ks4
#
# One sorter per process, so the peak RSS is that sorter's. Results go to
# `benchmarks/sorters/results/sorter_resources_<sorter>.json`; `sorter_resources_table.py` draws them.
# Build with `maturin develop --release` for timings.
#
# Data (local, git-ignored): `.data/kilosort4/ZFM-02370_mini.imec0.ap.short.bin` (+ `saved_results/`
# for the probe), `.data/nwb/15-25-33_meps.nwb.zarr` (HDEMG series).

# %% [1] Settings, Recording, Probe
import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

import dsp_kitchen as dk
import dsp_kitchen.synapse as syn
from dsp_kitchen.io import Recording, list_sources

DATA_DIR = Path(os.environ.get("DSP_KITCHEN_DATA", Path(__file__).resolve().parents[2] / ".data"))
KS4_BIN = DATA_DIR / "kilosort4" / "ZFM-02370_mini.imec0.ap.short.bin"
KS4_RESULTS = DATA_DIR / "kilosort4" / "saved_results"
NWB_PATH = DATA_DIR / "nwb" / "15-25-33_meps.nwb.zarr"
GRID_ROWS, GRID_COLS, ELECTRODE_PITCH_UM = 4, 8, 100.0
OUTPUT = Path(__file__).resolve().parent / "results"
#: Memory sampling period (s).
SAMPLE_SEC = 0.05

parser = argparse.ArgumentParser()
parser.add_argument("sorter", choices=["ks4", "emusort"])
parser.add_argument("--seconds", type=float, default=None, help="sort the first N s (default: all)")
parser.add_argument("--repeats", type=int, default=2, help="runs; the first is cold")
args = parser.parse_args()

HAS_SMI = shutil.which("nvidia-smi") is not None


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.split()[0])


def gpu_name():
    if not HAS_SMI:
        return None
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def cpu_name():
    with open("/proc/cpuinfo") as f:
        return next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), platform.machine())


def rss_mib():
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) / 1024 for line in f if line.startswith("VmRSS:"))


class PeakMemory:
    """Peak host RSS and GPU memory in use while the block runs, sampled in a thread."""

    def __init__(self):
        self.done = threading.Event()
        self.rss_mib = 0.0
        self.gpu_mib = 0.0
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.done.is_set():
            self.rss_mib = max(self.rss_mib, rss_mib())
            if HAS_SMI:
                self.gpu_mib = max(self.gpu_mib, gpu_used_mib())
            time.sleep(SAMPLE_SEC)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join()


gpu_idle_mib = gpu_used_mib() if HAS_SMI else None

if args.sorter == "ks4":
    from dsp_kitchen.synapse.ml import kilosort4 as sorter

    rec = Recording(str(KS4_BIN))
    probe = syn.ProbeLayout.from_recording(str(KS4_BIN)) or syn.load_sorting(str(KS4_RESULTS)).probe
    if rec.channels != probe.total_channels:
        rec = rec.slice_samples(channels=probe.channel_ids())
    config = sorter.Config()
    config.whitening_range = min(config.whitening_range, probe.total_channels)
    source = "Neuropixels 1.0"
else:
    from dsp_kitchen.synapse.ml import emusort as sorter

    hdemg = next(s["id"] for s in list_sources(str(NWB_PATH)) if "HDEMG" in s["id"])
    rec = Recording(str(NWB_PATH), source=hdemg)
    probe = syn.ProbeLayout.hdemg_grid("HD-EMG 4x8", GRID_ROWS, GRID_COLS, ELECTRODE_PITCH_UM)
    config = sorter.Config()
    source = f"HD-EMG {GRID_ROWS} × {GRID_COLS} grid"
if args.seconds is not None:
    rec = rec.slice_time(start_sec=0.0, end_sec=args.seconds)
fs = rec.sample_rate
duration_sec = rec.samples / fs
print(f"{rec}\n{probe}\nruntime {dk.runtime.current()}, GPU {gpu_name()}", flush=True)


# %% [2] Runs
def run():
    starts = {}

    def progress(stage, step, steps, done, total, unit):
        if stage not in starts:
            starts[stage] = time.perf_counter()
            print(f"  [{starts[stage] - t0:7.1f} s] {stage}", flush=True)

    t0 = time.perf_counter()
    with PeakMemory() as peak:
        result = sorter.run(rec, probe, config, progress=progress)
    wall = time.perf_counter() - t0
    # Each stage lasts until the next one reports; the first also counts the setup before it
    order = sorted(starts, key=starts.get)
    ends = [starts[s] for s in order[1:]] + [t0 + wall]
    stages = {s: e - starts[s] for s, e in zip(order, ends)}
    stages[order[0]] += starts[order[0]] - t0
    spikes = result.spikes()
    summary = {
        "wall_sec": wall,
        "stages_sec": stages,
        "peak_rss_mib": peak.rss_mib,
        "peak_gpu_mib": peak.gpu_mib - gpu_idle_mib if HAS_SMI else None,
        "spikes": len(spikes["sample"]),
        "units": int(result.n_units),
    }
    return summary, spikes


runs, outputs = [], []
for repeat in range(args.repeats):
    summary, spikes = run()
    runs.append(summary)
    outputs.append(spikes)
    gpu = "n/a" if summary["peak_gpu_mib"] is None else f"{summary['peak_gpu_mib'] / 1024:.2f} GiB"
    print(f"run {repeat + 1} ({'cold' if repeat == 0 else 'warm'}): {summary['wall_sec']:.1f} s "
          f"({duration_sec / summary['wall_sec']:.1f}× real time), RSS {summary['peak_rss_mib'] / 1024:.2f} GiB, "
          f"GPU {gpu}, {summary['spikes']:,} spikes, {summary['units']} units", flush=True)
reproducible = all(np.array_equal(o["sample"], outputs[0]["sample"]) and np.array_equal(o["unit"], outputs[0]["unit"])
                   for o in outputs[1:])
print(f"reproducible: {reproducible}")

# %% [3] Save
report = {
    "sorter": args.sorter,
    "source": source,
    "channels": rec.channels,
    "sample_rate": fs,
    "duration_sec": duration_sec,
    "signal_f32_gib": rec.samples * rec.channels * 4 / 1024**3,
    "runtime": str(dk.runtime.current()),
    "gpu": gpu_name(),
    "cpu": cpu_name(),
    "process_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "runs": runs,
    "reproducible": reproducible,
}
OUTPUT.mkdir(parents=True, exist_ok=True)
path = OUTPUT / f"sorter_resources_{args.sorter}.json"
path.write_text(json.dumps(report, indent=1))
print(f"saved {path}")
