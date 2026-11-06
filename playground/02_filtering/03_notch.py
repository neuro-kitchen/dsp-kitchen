# %% [markdown]
# # 02 · Filtering — 3. A notch for mains hum
# Mains hum (60 Hz, or 50 Hz in Europe) and its harmonics sit right inside the LFP band, so a
# low-pass cannot remove them. A notch takes out one frequency and leaves the rest: one per line
# (here 60 Hz and its third harmonic, 180 Hz). The synthetic recording knows its hum, so the result
# can be checked against the recording without it.

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic()
Q = 30.0  # width: the notch spans about freq / Q Hz
out = dspk.ops.filter.notch(s.data, s.sample_rate, 60.0, Q)
out = dspk.ops.filter.notch(out, s.sample_rate, 180.0, Q)

# How much of the hum is left: residual against the recording without hum
truth = s.without("hum")
left = np.std(out - truth) / np.std(s.components["hum"])
print(f"hum left after two notches: {100 * left:.1f} % of its RMS")

# %% Figure
fig, _ = figures.before_after(s.data, out, s.sample_rate, title="Two notches take the hum out, nothing else",
                              labels=("recorded", "after notches at 60 and 180 Hz"), fmax=2_000.0, duration=0.1)
plating.save(fig, FIG / "03_notch.png", note=f"dspk.ops.filter.notch at 60 and 180 Hz, Q = {Q:.0f}, zero phase; {100 * left:.1f} % of the hum's RMS is left.")
