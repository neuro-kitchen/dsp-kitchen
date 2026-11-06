# %% [markdown]
# # 06 · Pipelines — 1. A chain of stages on the device
# Calling `dspk.ops` functions one after the other moves the data to the device and back at every
# step. A `dspk.ops.Pipeline` keeps it there: one upload, every stage, one download, with filter
# designs and device buffers reused from call to call. On a GPU most of a single call's time is the
# transfer, so a chain gains the most.

# %%
from pathlib import Path
import time

import numpy as np
from scipy import signal as sp

import dspk
import plating
from dspk.ops import Pipeline, stage
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic(channels=64, seconds=10.0)
BAND = (300.0, 6_000.0)


def one_by_one(x):
    y = dspk.ops.filter.notch(x, s.sample_rate, 60.0, 30.0)
    y = dspk.ops.filter.iir(y, s.sample_rate, "bandpass", BAND)
    return dspk.ops.spatial.cmr(y)


pipe = Pipeline([stage.notch(60.0, 30.0), stage.iir("bandpass", BAND), stage.cmr()], s.sample_rate)
sos_notch = sp.tf2sos(*sp.iirnotch(60.0, 30.0, fs=s.sample_rate))
sos_band = sp.butter(5, BAND, "bandpass", fs=s.sample_rate, output="sos")


def scipy_chain(x):
    pad = dspk.ops.filter.settling_samples
    y = sp.sosfiltfilt(sos_notch, x, axis=1, padtype="odd", padlen=pad(sos_notch))
    y = sp.sosfiltfilt(sos_band, y, axis=1, padtype="odd", padlen=pad(sos_band))
    return y - np.median(y, axis=0, keepdims=True)


def timed(f):
    f(s.data)  # warm-up: compiles and tunes the kernels (cached on disk for later runs)
    start = time.perf_counter()
    out = f(s.data)
    return (time.perf_counter() - start) * 1e3, out


cases = {"scipy, on the CPU": timed(scipy_chain), "dspk, one call per stage": timed(one_by_one), "dspk, one Pipeline": timed(pipe.process)}
ref = cases["scipy, on the CPU"][1]
for name, (ms, out) in cases.items():
    print(f"{name:26s} {ms:8.1f} ms   max difference from scipy {np.max(np.abs(out - ref)) / np.max(np.abs(ref)):.1e} of the peak")

# %% Figure: time of each way, the same result
p = plating.theme.palette()
names = list(cases)
fig, ax = plating.figure("The same chain, three ways", size=(11.0, 3.2))
for i, (name, color) in enumerate(zip(names, (p.muted, p.series[1], p.series[0]))):
    ms = cases[name][0]
    ax.barh(i, ms, height=0.55, color=color)
    ax.text(ms + 15, i, figures.ms_label(ms), va="center", fontsize=8.5, color=p.ink)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlim(0, max(c[0] for c in cases.values()) * 1.2)
ax.grid(axis="y", visible=False)
ax.set_xlabel("milliseconds")
ax.set_title(f"notch 60 Hz → band-pass {BAND[0]:.0f}–{BAND[1]:.0f} Hz → CMR on {s.channels} channels × {s.samples / s.sample_rate:.0f} s; same output (≤ 1e-5)")
plating.save(fig, FIG / "01_chain_on_device.png", note="Times from a NumPy array to a NumPy array, after a warm-up call.")
