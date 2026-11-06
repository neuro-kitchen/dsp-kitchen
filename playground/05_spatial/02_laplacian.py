# %% [markdown]
# # 05 · Spatial — 2. A surface Laplacian
# On a grid of electrodes (HD-EMG, ECoG), a *Laplacian* replaces each channel by itself minus the
# mean of its neighbours. Whatever is spread over the whole grid (a distant source, the common
# artefact) cancels; what is local to a few electrodes stays. `Laplacian.from_grid(rows, cols)` is
# the 5-point version; `from_positions` handles any layout.

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
ROWS, COLS = 4, 4
# No LFP here (its phase differs across channels, so it is not shared): the shared parts are the
# common artefact and the hum, made large
s = signals.synthetic(channels=ROWS * COLS, lfp_uv=0.0, common_uv=150.0, hum_uv=60.0)
lap = dspk.ops.spatial.Laplacian.from_grid(ROWS, COLS)
y = lap.apply(s.data)

ch = s.loudest_channel
for name, comp in (("common artefact", "common"), ("spikes", "spikes")):
    kept = np.std(lap.apply(s.components[comp])[ch]) / (np.std(s.components[comp][ch]) or 1)
    print(f"{name}: {100 * kept:.0f} % of its RMS kept on channel {ch}")

# %% Figure: the grid's channels before and after (the same scale)
p = plating.theme.palette()
fig, (left, right) = plating.figure(f"A Laplacian on a {ROWS} × {COLS} grid keeps what is local", cols=2, size=(11.0, 7.0), sharey=True)
spacing = figures.traces(left, s.data, s.sample_rate, duration=0.4, color=p.muted)
left.set_title("recorded: a large wave and the hum, shared by every channel")
figures.traces(right, y, s.sample_rate, duration=0.4, spacing=spacing)
right.set_title("after the Laplacian (same scale): the shared parts cancel; the spikes stay")
plating.save(fig, FIG / "02_laplacian.png", note=f"dspk.ops.spatial.Laplacian.from_grid({ROWS}, {COLS}) on the synthetic recording (16 channels).")
