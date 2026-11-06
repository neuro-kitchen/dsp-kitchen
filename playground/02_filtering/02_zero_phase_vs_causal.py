# %% [markdown]
# # 02 · Filtering — 2. Zero phase or causal
# A filter run forward only (*causal*, as hardware does it live) delays and reshapes what it passes.
# Run forward and then backward (*zero phase*, `mode="forward_backward"`, the default) the delays
# cancel: a spike stays where it was, with its shape. Causal filters are for streams that cannot
# wait for the future; zero phase for everything recorded. dspk's zero-phase result equals scipy's
# `sosfiltfilt`.

# %%
from pathlib import Path

import numpy as np
from scipy import signal as sp

import dspk
import plating
from plating import signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic(noise_uv=2.0)  # quieter, so the spike's shape is easy to follow
BAND = (300.0, 6_000.0)
zero = dspk.ops.filter.iir(s.data, s.sample_rate, "bandpass", BAND)
causal = dspk.ops.filter.iir(s.data, s.sample_rate, "bandpass", BAND, mode="forward")

# The same zero-phase filter in scipy (same padding)
sos = sp.butter(5, BAND, "bandpass", fs=s.sample_rate, output="sos")
ref = sp.sosfiltfilt(sos, s.data.astype(np.float64), axis=1, padtype="odd", padlen=dspk.ops.filter.settling_samples(sos))
print(f"zero phase vs scipy sosfiltfilt: max difference {np.max(np.abs(zero - ref)) / np.max(np.abs(ref)):.1e} of the peak")

# %% Figure: one spike of the largest unit, as put in and after each filter
p = plating.theme.palette()
ch = s.loudest_channel
unit = int(np.argmax(np.abs(s.templates).max(axis=1)))
at = s.spike_samples[s.spike_units == unit][1]
half = int(0.003 * s.sample_rate)
win = slice(at - half, at + half)
t = (np.arange(-half, half)) / s.sample_rate * 1e3
peak = lambda x: t[np.argmin(x)]
fig, ax = plating.figure("Zero phase keeps a spike's shape and place; causal reshapes and shifts it", size=(11.0, 4.8))
ax.plot(t, s.components["spikes"][ch, win], color=p.muted, linewidth=2.6, label=f"the spike put in (trough at {peak(s.components['spikes'][ch, win]):+.2f} ms)")
ax.plot(t, zero[ch, win], color=p.series[0], linewidth=1.4, label=f"zero phase (trough at {peak(zero[ch, win]):+.2f} ms)")
ax.plot(t, causal[ch, win], color=p.series[1], linewidth=1.4, label=f"causal (trough at {peak(causal[ch, win]):+.2f} ms)")
ax.axvline(0, color=p.ink2, linewidth=0.8, linestyle=(0, (4, 3)))
ax.set_xlabel("time from the spike (ms)")
ax.set_ylabel("µV")
ax.set_xlim(t[0], t[-1])
ax.set_title(f"band-pass {BAND[0]:.0f}–{BAND[1]:.0f} Hz, Butterworth order 5, channel {ch}")
ax.legend(loc="lower right")
plating.save(fig, FIG / "02_zero_phase_vs_causal.png", note="dspk.ops.filter.iir(mode='forward_backward' | 'forward'); zero phase equals scipy.signal.sosfiltfilt.")
