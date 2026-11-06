# %% [markdown]
# # 05 · Spatial — 1. Whitening
# Neighbouring channels see much of the same signal, so they are correlated. *Whitening* is a
# channels × channels matrix that undoes that: after it, every channel has unit variance and no
# correlation with the others. Spike sorters (Kilosort) whiten so that each channel's evidence
# counts once. `Whitening.fit_zca` fits the matrix on the device (`fit_local_knn`: over each
# channel's nearest neighbours only, as Kilosort4).

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic(channels=16)
x = dspk.ops.filter.iir(s.data, s.sample_rate, "bandpass", (300.0, 6_000.0))  # the spike band, as sorters whiten it
# Make neighbours share signal: each channel picks up a fraction of the next ones (volume conduction)
mix = np.eye(s.channels) + sum(0.6**k * (np.eye(s.channels, k=k) + np.eye(s.channels, k=-k)) for k in (1, 2, 3))
x = (mix @ x).astype(np.float32)

w = dspk.ops.spatial.Whitening.fit_zca(x)
y = w.apply(x)
before, after = np.corrcoef(x), np.corrcoef(y)
off = ~np.eye(s.channels, dtype=bool)
print(f"largest correlation between two channels: before {np.max(np.abs(before[off])):.2f}, after {np.max(np.abs(after[off])):.3f}")

# %% Figure: the channels' correlation matrices
fig, (left, right) = plating.figure("Whitening removes the correlation between channels", cols=2, size=(11.0, 5.0))
figures.matrix(left, before, colorbar_label="correlation")
left.set_title("before: neighbours share signal")
figures.matrix(right, after, colorbar_label="correlation")
right.set_title("after: each channel on its own")
for ax in (left, right):
    ax.set_xlabel("channel")
    ax.set_ylabel("channel")
plating.save(fig, FIG / "01_whitening.png", note="dspk.ops.spatial.Whitening.fit_zca on 16 channels of the spike band, mixed with their neighbours.")
