# %% [markdown]
# # dspk vs scipy: time and agreement per signal operation
# One recording-sized block (384 channels × 60 000 samples, 2 s of Neuropixels at 30 kHz, float32)
# through each operation, on every runtime compiled in, against scipy / numpy on the host:
# zero-phase and causal band-pass, notch, FIR, median filter, polyphase resampling, common median
# reference. dspk's time is end to end from a NumPy array (upload, kernel, download), the time a
# Python caller sees; the kernel alone is in `benchmarks/kernels/tests/bench_kernels.rs`.
#
#     uv run python benchmarks/signal-operations/scipy_timing.py [--repeats 5] [--channels 384] [--samples 60000]
#
# Writes `benchmarks/signal-operations/results/scipy_timing-<runtime>.json` (plating's record: machine,
# versions, per operation the times, the error and pass / fail) and its two figures.
# Timings are medians after one warm-up (the warm-up compiles and autotunes the kernels).

# %% Settings
import argparse
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import signal as sp

import dspk
import plating
from plating import figures, results

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
#: Largest error each operation may have against scipy, relative to scipy's peak: float32 on the
#: device against scipy's float64 for the filters; median and CMR pick the same samples (exact)
TOLERANCE = {"zero-phase band-pass": 1e-5, "causal band-pass": 1e-5, "notch 60 Hz": 1e-5, f"FIR {FIR_TAPS} taps": 1e-5,
             f"median width {MEDIAN_WIDTH}": 1e-6, f"resample {UP}/{DOWN}": 1e-5, "common median reference": 1e-6}
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
cases = {rt: [] for rt in runtimes}
for name, (ours, theirs) in OPERATIONS.items():
    t_ref, ref = timed(theirs)
    cells, errors = "", []
    for rt in runtimes:
        t_ours, y = timed(lambda: ours(rt))
        case = results.compare(name, y, ref, tolerance=TOLERANCE[name], dspk_ms=t_ours * 1e3, reference_ms=t_ref * 1e3)
        cases[rt].append(case)
        cells += f" {t_ours * 1e3:7.1f}ms"
        errors.append(f"{case.rel_err:.0e}{'' if case.passed else ' FAIL'}")
    print(f"{name:26s} {t_ref * 1e3:7.1f}ms{cells}   {' '.join(errors)}")

# %% Save: one record per runtime (plating.results schema), and its two figures
NOTES = (f"One {args.channels} × {args.samples} float32 block (2 s of Neuropixels at 30 kHz). Each time is the median of "
         f"{args.repeats} calls after a warm-up, end to end from a NumPy array (upload, kernels, download) for dspk; "
         f"scipy / numpy in float64 on the CPU. Error: max |dspk − scipy| / max |scipy|.")
for rt in runtimes:
    path = results.write(RESULTS / f"scipy_timing-{rt}.json", "signal-operations/scipy", cases[rt],
                         reference={"tool": "scipy", "version": scipy.__version__}, runtime=rt, notes=NOTES)
    plating.save(figures.timings(path, title=f"dspk ({rt}) against scipy, {args.channels} channels × 2 s")[0], RESULTS / f"scipy_timing-{rt}.png", note=NOTES.split(". ")[0] + ".", runtime=rt)
    plating.save(figures.agreement(path, title=f"dspk ({rt}) agrees with scipy")[0], RESULTS / f"scipy_agreement-{rt}.png", runtime=rt)
    print("record:", path)
