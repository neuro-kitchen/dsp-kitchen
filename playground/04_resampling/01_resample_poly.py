# %% [markdown]
# # 04 · Resampling — 1. Fewer samples, the same slow signal
# LFP analyses do not need 30 000 samples a second. `resample_poly(x, up, down)` changes the rate
# by `up / down` behind an anti-alias low-pass (polyphase FIR, Kaiser window, as scipy's
# `resample_poly`); `decimate(x, q)` divides it by `q` behind an IIR low-pass, as scipy's `decimate`.
# Both run on the device.

# %%
from pathlib import Path

import numpy as np
from scipy import signal as sp

import dspk
import plating
from plating import signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic()
DOWN = 30  # 30 kHz → 1 kHz
low = dspk.ops.resample.resample_poly(s.data, 1, DOWN)
dec = dspk.ops.resample.decimate(s.data, 10)  # q ≤ 13 per step, as scipy advises
dec = dspk.ops.resample.decimate(dec, 3)
ref = sp.resample_poly(s.data.astype(np.float64), 1, DOWN, axis=1, window=("kaiser", 5.0))
print(f"{s.samples:,} → {low.shape[1]:,} samples per channel")
print(f"resample_poly vs scipy: max difference {np.max(np.abs(low - ref)) / np.max(np.abs(ref)):.1e} of the peak")

# %% Figure: the slow signals at 30 kHz and at 1 kHz
p = plating.theme.palette()
ch = s.loudest_channel
slow = s.components["lfp"][ch] + s.components["common"][ch] + s.components["hum"][ch]
fs_low = s.sample_rate / DOWN
fig, ax = plating.figure(f"From {s.sample_rate / 1e3:.0f} kHz to {fs_low / 1e3:.0f} kHz: the slow signals stay", size=(11.0, 4.6))
ax.plot(s.time, s.data[ch], color=p.muted, linewidth=0.6, label=f"recorded, {s.sample_rate / 1e3:.0f} kHz")
ax.plot(s.time, slow, color=p.ink2, linewidth=1.0, linestyle=(0, (4, 3)), label="LFP + common + hum (put in)")
ax.plot(np.arange(low.shape[1]) / fs_low, low[ch], color=p.series[0], linewidth=1.4, label=f"resample_poly 1/{DOWN}, {fs_low / 1e3:.0f} kHz")
ax.plot(np.arange(dec.shape[1]) / fs_low, dec[ch], color=p.series[1], linewidth=1.0, label="decimate 10 then 3")
ax.set_xlabel("time (s)")
ax.set_ylabel("µV")
ax.set_xlim(0, 1.0)
ax.set_title(f"channel {ch}, first second: everything below {fs_low / 2:.0f} Hz stays (the 60 Hz hum too); the spikes and the fast noise go")
ax.legend(loc="lower left", ncols=2)
plating.save(fig, FIG / "01_resample_poly.png", note="dspk.ops.resample.resample_poly (Kaiser β 5, equal to scipy) and decimate (Chebyshev I, zero phase).")
