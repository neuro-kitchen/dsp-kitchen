# %% [markdown]
# # dspk: IIR filters on the device, against scipy
# `dspk.ops.filter` designs second-order sections exactly as `scipy.signal` does (float64) and runs
# them on the chosen runtime (float32): zero-phase like `sosfiltfilt`, causal like `sosfilt`.
# This compares both, on every runtime compiled in, and plots the result.

# %% Imports and a synthetic signal
from pathlib import Path

import numpy as np
from scipy import signal as sp

import dspk

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)
FS = 30_000.0
rng = np.random.default_rng(0)
t = np.arange(30_000) / FS
# 4 channels: 60 Hz hum, a 1 kHz tone, broadband noise and a DC offset
x = np.stack([20 * np.sin(2 * np.pi * 60 * t) + 5 * np.sin(2 * np.pi * 1_000 * t) + rng.standard_normal(t.size) + 3.0 for _ in range(4)])
x = x.astype(np.float32)
print("dspk", dspk.__version__, "| runtimes:", dspk.runtime.available())

# %% Designs: identical sections to scipy
ours = dspk.ops.filter.sos("bandpass", (300.0, 6_000.0), FS)
theirs = sp.butter(5, (300.0, 6_000.0), "bandpass", fs=FS, output="sos")
print("bandpass sections, max difference vs scipy:", np.max(np.abs(ours - theirs)))

# %% Filtering on every runtime vs scipy (float64)
pad = min(dspk.ops.filter.settling_samples(theirs), x.shape[1] - 1)
reference = {
    "zero-phase bandpass": sp.sosfiltfilt(theirs, x.astype(np.float64), axis=1, padtype="odd", padlen=pad),
    "causal bandpass": sp.sosfilt(theirs, x.astype(np.float64), axis=1),
}
peak = max(np.max(np.abs(r)) for r in reference.values())
results = {}
for runtime in dspk.runtime.available():
    try:
        results[runtime] = {
            "zero-phase bandpass": dspk.ops.filter.iir(x, FS, "bandpass", (300.0, 6_000.0), runtime=runtime),
            "causal bandpass": dspk.ops.filter.iir(x, FS, "bandpass", (300.0, 6_000.0), mode="forward", runtime=runtime),
        }
    except RuntimeError as e:  # a runtime compiled in without a device on this machine
        print(f"{runtime}: skipped ({e})")
        continue
    for name, y in results[runtime].items():
        print(f"{runtime:5s} {name:20s} max error {np.max(np.abs(y - reference[name])) / peak:.1e} of the peak")

# %% Notch: removes the 60 Hz hum
clean = dspk.ops.filter.notch(x, FS, 60.0, 30.0)
f, p_in = sp.welch(x[0], FS, nperseg=8_192)
_, p_out = sp.welch(clean[0], FS, nperseg=8_192)
print(f"60 Hz power before / after notch: {p_in[np.argmin(np.abs(f - 60))]:.3g} / {p_out[np.argmin(np.abs(f - 60))]:.3g}")

# %% Plots
if plt is not None:
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))
    w, h = sp.sosfreqz(theirs, worN=4_096, fs=FS)
    ax[0].semilogx(w[1:], 20 * np.log10(np.abs(h[1:])), label="bandpass 300–6000 Hz (one pass)")
    ax[0].set(xlabel="Hz", ylabel="dB", ylim=(-80, 5), title="Design (identical to scipy)")
    ax[0].legend()
    window = slice(10_000, 10_600)
    for runtime, ys in results.items():
        ax[1].plot(t[window], ys["zero-phase bandpass"][0, window], label=f"dspk {runtime}", lw=1)
    ax[1].plot(t[window], reference["zero-phase bandpass"][0, window], "k--", lw=1, label="scipy sosfiltfilt")
    ax[1].set(xlabel="s", title="Zero-phase bandpass, channel 0")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "base_01_filters_vs_scipy.png", dpi=150)
    print("figure:", OUT / "base_01_filters_vs_scipy.png")
