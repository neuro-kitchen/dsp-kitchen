# %% [markdown]
# # 02 · Filtering — 1. Low-pass and high-pass
# One cutoff splits a recording in two: below ~300 Hz the slow field potentials (LFP), above it
# the spikes. `dspk.ops.filter.iir` designs the filter for the recording's sample rate and runs it
# on the device (Butterworth, order 5, zero phase by default).

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from plating import signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic()
CUTOFF = 300.0
low = dspk.ops.filter.iir(s.data, s.sample_rate, "lowpass", CUTOFF)
high = dspk.ops.filter.iir(s.data, s.sample_rate, "highpass", CUTOFF)

# What each should keep: everything slow (LFP, the common artefact, the 60 Hz hum) or the spikes
slow = s.components["lfp"] + s.components["common"] + s.components["hum"]
ch = s.loudest_channel
for name, got, want in (("low-pass", low, slow), ("high-pass", high, s.components["spikes"])):
    r = np.corrcoef(got[ch], want[ch])[0, 1]
    print(f"{name}: correlation with what it should keep {r:.3f}")

# %% Figure: the recording, then each half beside what it should be
p = plating.theme.palette()
win = slice(0, int(0.4 * s.sample_rate))
t = s.time[win]
fig, axes = plating.figure(f"One cutoff, two signals: below and above {CUTOFF:.0f} Hz", rows=3, size=(11.0, 7.4), sharex=True)
axes[0].plot(t, s.data[ch, win], color=p.series[0], linewidth=0.9)
axes[0].set_title(f"recorded (channel {ch})")
axes[1].plot(t, slow[ch, win], color=p.muted, linewidth=2.4, label="LFP + common + hum (what was put in)")
axes[1].plot(t, low[ch, win], color=p.series[0], linewidth=1.0, label=f"low-pass {CUTOFF:.0f} Hz")
axes[1].set_title("below the cutoff: the slow signals")
axes[1].legend(loc="upper right", ncols=2)
axes[2].plot(t, high[ch, win], color=p.series[0], linewidth=0.8, label=f"high-pass {CUTOFF:.0f} Hz")
# Drawn over the noisy result, so the spikes put in stay visible
axes[2].plot(t, s.components["spikes"][ch, win], color=p.ink2, linewidth=1.2, label="spikes (what was put in)", zorder=3)
axes[2].set_title("above the cutoff: the spikes (and the noise)")
axes[2].legend(loc="lower right", ncols=2)
for ax in axes:
    ax.set_ylabel("µV")
axes[-1].set_xlabel("time (s)")
axes[-1].set_xlim(t[0], t[-1])
plating.save(fig, FIG / "01_lowpass_highpass.png", note=f"dspk.ops.filter.iir: Butterworth order 5, zero phase, {CUTOFF:.0f} Hz; synthetic recording.")
