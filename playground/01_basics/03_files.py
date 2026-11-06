# %% [markdown]
# # 01 · Basics — 3. Files in and out
# `dspk.io.write_raw` writes a raw binary with a JSON `.meta` beside it (any stored type: here
# 16-bit integers with a gain, as acquisition systems store them); `dspk.io.write_zarr` writes a
# compressed Zarr store. `dspk.io.open` reads either (and SpikeGLX, NWB) and returns values in their
# unit, whatever the stored type.

# %%
from pathlib import Path
import tempfile

import numpy as np

import dspk
import plating
from plating import signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic()
work = Path(tempfile.mkdtemp(prefix="dspk-playground-"))

# int16 with a gain of 0.195 µV per step (Intan's), and float32 Zarr
GAIN = 0.195
dspk.io.write_raw(s.data, work / "recording.bin", s.sample_rate, format="int16", gain=GAIN, unit="µV")
dspk.io.write_zarr(s.data, work / "recording.zarr", s.sample_rate, unit="µV")
raw, zarr = dspk.io.open(work / "recording.bin"), dspk.io.open(work / "recording.zarr")
print(raw)
print(zarr)

# %% What comes back: Zarr exactly, int16 within half a step
from_raw, from_zarr = raw.read(), zarr.read()
print("zarr  max difference:", np.max(np.abs(from_zarr - s.data)))
print("int16 max difference:", np.max(np.abs(from_raw - s.data)), f"(half a step: {GAIN / 2:.4f} µV)")
assert np.array_equal(from_zarr, s.data) and np.max(np.abs(from_raw - s.data)) <= GAIN / 2 + 1e-4

# %% Figure: a stretch of channel 0 as written and as read from int16, and the difference
p = plating.theme.palette()
fig, (top, bottom) = plating.figure("Writing and reading a recording", rows=2, size=(11.0, 6.0), sharex=True, height_ratios=[2, 1])
n = int(0.01 * s.sample_rate)
t = s.time[:n] * 1e3
top.plot(t, s.data[0, :n], color=p.muted, linewidth=2.2, label="written (float32)")
top.plot(t, from_raw[0, :n], color=p.series[0], linewidth=1.0, label="read back from int16")
top.set_ylabel("µV")
top.legend(loc="upper right", ncols=2)
top.set_title("channel 0, first 10 ms")
bottom.plot(t, from_raw[0, :n] - s.data[0, :n], color=p.series[0], linewidth=1.0)
for y in (GAIN / 2, -GAIN / 2):
    bottom.axhline(y, color=p.ink2, linewidth=1, linestyle=(0, (4, 3)))
bottom.set_ylabel("µV")
bottom.set_xlabel("time (ms)")
bottom.set_title(f"difference: rounding to int16 steps of {GAIN} µV stays within ± half a step (dashed)")
plating.save(fig, FIG / "03_files.png", note="write_raw(format='int16', gain=0.195) and write_zarr (float32), read back with dspk.io.open.")
