# %% [markdown]
# # dspk vs scipy: time and agreement per signal operation
# One recording-sized block (384 channels × 60 000 samples, 2 s of Neuropixels at 30 kHz, float32)
# through each operation, on every runtime compiled in, against scipy / numpy on the host:
# zero-phase and causal band-pass, notch, FIR, median filter, polyphase resampling, common median
# reference. dspk's time is end to end from a NumPy array (upload, kernel, download), the time a
# Python caller sees; the kernel alone is in `crates/dspk-ops/tests/bench_kernels.rs`.
#
#     uv run python benchmarks/signal-operations/scipy_timing.py [--repeats 5] [--channels 384] [--samples 60000]
#
# Writes `benchmarks/signal-operations/results/scipy_timing.json` (with the machine and versions).
# Timings are medians after one warm-up (the warm-up compiles and autotunes the kernels).

# %% Settings
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import signal as sp

import dspk

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repeats", type=int, default=5, help="timed runs per operation (after one warm-up)")
parser.add_argument("--channels", type=int, default=384)
parser.add_argument("--samples", type=int, default=60_000)
args, _ = parser.parse_known_args()

RESULTS = Path(__file__).resolve().parent / "results"
FS = 30_000.0
BAND = (300.0, 6_000.0)
NOTCH_HZ, NOTCH_Q = 60.0, 30.0
FIR_TAPS = 101
MEDIAN_WIDTH = 5
UP, DOWN = 1, 3

rng = np.random.default_rng(0)
t = np.arange(args.samples) / FS
x = (rng.standard_normal((args.channels, args.samples)) * 10 + 20 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)

# %% Operations: (dspk call, scipy / numpy call), both from the same float32 array
sos = sp.butter(5, BAND, "bandpass", fs=FS, output="sos")
pad = dspk.ops.filter.settling_samples(sos)
b_notch, a_notch = sp.iirnotch(NOTCH_HZ, NOTCH_Q, fs=FS)
taps = dspk.ops.filter.firwin(FIR_TAPS, 3_000.0, sample_rate=FS)
OPERATIONS = {
    "zero-phase band-pass": (
        lambda rt: dspk.ops.filter.iir(x, FS, "bandpass", BAND, runtime=rt),
        lambda: sp.sosfiltfilt(sos, x, axis=1, padtype="odd", padlen=pad),
    ),
    "causal band-pass": (
        lambda rt: dspk.ops.filter.iir(x, FS, "bandpass", BAND, mode="forward", runtime=rt),
        lambda: sp.sosfilt(sos, x, axis=1),
    ),
    "notch 60 Hz": (
        lambda rt: dspk.ops.filter.notch(x, FS, NOTCH_HZ, NOTCH_Q, runtime=rt),
        lambda: sp.filtfilt(b_notch, a_notch, x, axis=1, padtype="odd", padlen=dspk.ops.filter.settling_samples(sp.tf2sos(b_notch, a_notch))),
    ),
    f"FIR {FIR_TAPS} taps": (
        lambda rt: dspk.ops.filter.fir(x, taps, runtime=rt),
        lambda: sp.lfilter(taps, 1.0, x, axis=1),
    ),
    f"median width {MEDIAN_WIDTH}": (
        lambda rt: dspk.ops.filter.median(x, MEDIAN_WIDTH, runtime=rt),
        lambda: sp.medfilt(x, (1, MEDIAN_WIDTH)),
    ),
    f"resample {UP}/{DOWN}": (
        lambda rt: dspk.ops.resample.resample_poly(x, UP, DOWN, runtime=rt),
        lambda: sp.resample_poly(x, UP, DOWN, axis=1, window=("kaiser", 5.0)),
    ),
    "common median reference": (
        lambda rt: dspk.ops.spatial.cmr(x, runtime=rt),
        lambda: x - np.median(x, axis=0, keepdims=True),
    ),
}


def timed(call):
    """Median wall time (s) of `args.repeats` calls after one warm-up, and the last result."""
    out = call()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        out = call()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), out


# %% Run
runtimes = []
for rt in dspk.runtime.available():
    try:
        dspk.ops.spatial.car(x[:2, :16], runtime=rt)
        runtimes.append(rt)
    except RuntimeError as e:  # compiled in, no device on this machine
        print(f"{rt}: skipped ({e})")
print(f"dspk {dspk.__version__}, scipy {scipy.__version__} | {args.channels} ch × {args.samples} samples | runtimes {runtimes}\n")
header = f"{'operation':26s} {'scipy':>9s}" + "".join(f" {rt:>9s}" for rt in runtimes) + "   max error vs scipy (of its peak)"
print(header + "\n" + "-" * len(header))
rows = []
for name, (ours, theirs) in OPERATIONS.items():
    t_ref, ref = timed(theirs)
    peak = float(np.max(np.abs(ref))) or 1.0
    row = {"operation": name, "scipy_s": t_ref, "runtimes": {}}
    for rt in runtimes:
        t_ours, y = timed(lambda: ours(rt))
        row["runtimes"][rt] = {"seconds": t_ours, "speedup": t_ref / t_ours, "max_error": float(np.max(np.abs(y - ref))) / peak}
    rows.append(row)
    cells = "".join(f" {row['runtimes'][rt]['seconds'] * 1e3:7.1f}ms" for rt in runtimes)
    errors = " ".join(f"{row['runtimes'][rt]['max_error']:.0e}" for rt in runtimes)
    print(f"{name:26s} {t_ref * 1e3:7.1f}ms{cells}   {errors}")

# %% Save
RESULTS.mkdir(parents=True, exist_ok=True)
report = {
    "dspk": dspk.__version__,
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "machine": {"system": platform.platform(), "processor": platform.processor() or platform.machine()},
    "shape": [args.channels, args.samples],
    "repeats": args.repeats,
    "results": rows,
}
(RESULTS / "scipy_timing.json").write_text(json.dumps(report, indent=2))
print("\nreport:", RESULTS / "scipy_timing.json")
