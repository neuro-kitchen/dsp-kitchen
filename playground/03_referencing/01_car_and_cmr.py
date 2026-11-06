# %% [markdown]
# # 03 · Referencing — 1. Common average and common median
# Whatever every channel picks up alike (mains hum, movement, the reference electrode's own
# activity) can be estimated from the channels themselves and subtracted, at every sample:
# **CAR** subtracts the mean over channels, **CMR** the median. They agree on clean data; they part
# when one channel goes bad: the mean drags that channel's artefact onto all the others, the median
# ignores it.

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from plating import signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic(channels=16)
data = s.data.copy()

# One bad channel: a large, brief artefact on channel 3 (a loose contact)
BAD, AT, WIDTH, SIZE = 3, int(0.25 * s.sample_rate), int(0.02 * s.sample_rate), 2_000.0
data[BAD, AT : AT + WIDTH] += SIZE * np.hanning(WIDTH)

car = dspk.ops.spatial.car(data)
cmr = dspk.ops.spatial.cmr(data)

# What the bad channel did to a good one: referenced with it minus referenced without it
good = 0
for name, op, out in (("CAR", dspk.ops.spatial.car, car), ("CMR", dspk.ops.spatial.cmr, cmr)):
    leak = np.max(np.abs(out[good] - op(s.data)[good]))
    print(f"{name}: the bad channel's artefact reaches channel {good} with {leak:.0f} µV (it is {SIZE:,.0f} µV)")

# %% Figure: a good channel before, after CAR and after CMR, around the bad channel's artefact
p = plating.theme.palette()
win = slice(AT - 3 * WIDTH, AT + 4 * WIDTH)
t = s.time[win] * 1e3
fig, (top, bottom) = plating.figure("One bad channel: the mean spreads it, the median does not", rows=2, size=(11.0, 6.4), sharex=True)
top.plot(t, data[BAD, win], color=p.critical, linewidth=1.0, label=f"channel {BAD} (bad: a {SIZE:,.0f} µV artefact)")
top.plot(t, data[good, win], color=p.muted, linewidth=1.0, label=f"channel {good} (good)")
top.set_title("recorded")
top.legend(loc="upper right", ncols=2)
top.set_ylabel("µV")
bottom.plot(t, car[good, win], color=p.series[1], linewidth=1.2, label="after CAR (mean)")
bottom.plot(t, cmr[good, win], color=p.series[0], linewidth=1.2, label="after CMR (median)")
bottom.set_title(f"channel {good} after referencing: CAR carries 1/{s.channels} of the artefact, CMR none")
bottom.legend(loc="upper left", ncols=2)
bottom.set_ylabel("µV")
bottom.set_xlabel("time (ms)")
bottom.set_xlim(t[0], t[-1])
plating.save(fig, FIG / "01_car_and_cmr.png", note=f"dspk.ops.spatial.car / cmr over {s.channels} channels; both also remove the shared hum and artefact.")
