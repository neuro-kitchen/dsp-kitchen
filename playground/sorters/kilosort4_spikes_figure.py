# %% [markdown]
# # Kilosort4 spikes on the Kilosort4 test recording
#
# Waveforms of the largest *good* units and a raster over the preprocessed trace, from:
# - the sort: `dspk sort kilosort4 <recording> --probe neuropixels1 -o <phy folder>`;
# - the trace: the example pipeline (`playground/base/pipelines/neuropixels_preprocess.json`:
#   band-pass 300–6000 Hz zero phase + common median reference), written as raw or Zarr.
#
# Usage: `python kilosort4_spikes_figure.py <phy folder> <preprocessed recording> <out.png>`

# %% Settings
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dspk

PHY, TRACE, OUT = (Path(p) for p in sys.argv[1:4])
N_UNITS = 6  # largest good units shown
N_TRACES = 60  # individual spikes drawn per unit
HALF_WINDOW_MS = 1.5  # waveform window around each spike
RASTER_START_S, RASTER_LEN_S = 10.0, 0.3  # stretch of the raster
RASTER_CHANNELS = 24  # channels around the busiest unit's peak channel
CHANNEL_GAP_UV = 120.0  # vertical spacing of the raster traces

# %% Sort and trace
times = np.load(PHY / "spike_times.npy").ravel()
clusters = np.load(PHY / "spike_clusters.npy").ravel()
info = np.genfromtxt(PHY / "cluster_info.tsv", delimiter="\t", names=True, dtype=None, encoding="utf-8")
rec = dspk.io.open(TRACE)
fs = rec.sample_rate
half = int(HALF_WINDOW_MS * fs / 1000)

good = [row for row in info if row["group"] == "good"]
counts = {row["cluster_id"]: int(np.sum(clusters == row["cluster_id"])) for row in good}
units = sorted(counts, key=counts.get, reverse=True)[:N_UNITS]
peak = {row["cluster_id"]: int(row["ch"]) for row in info}
colours = plt.cm.tab10(np.linspace(0, 1, 10))

fig = plt.figure(figsize=(14, 9), constrained_layout=True)
grid = fig.add_gridspec(2, N_UNITS, height_ratios=[1, 1.6])

# %% (a) Waveforms on each unit's peak channel
t_ms = np.arange(-half, half) / fs * 1000
for i, u in enumerate(units):
    ax = fig.add_subplot(grid[0, i])
    spikes = times[clusters == u]
    spikes = spikes[(spikes > half) & (spikes < rec.samples - half)]
    pick = spikes[np.linspace(0, len(spikes) - 1, min(N_TRACES, len(spikes))).astype(int)]
    ch = peak[u]
    snippets = np.stack([rec.read(channels=[ch], start=int(s) - half, stop=int(s) + half)[0] for s in pick])
    ax.plot(t_ms, snippets.T, color=colours[i % 10], alpha=0.12, lw=0.6)
    ax.plot(t_ms, snippets.mean(axis=0), color="k", lw=1.6)
    ax.set_title(f"unit {u} · ch {ch}\n{counts[u]} spikes", fontsize=9)
    ax.set_xlabel("ms")
    if i == 0:
        ax.set_ylabel("µV (band-pass + CMR)")

# %% (b) Raster over the preprocessed trace
ax = fig.add_subplot(grid[1, :])
centre = peak[units[0]]
chans = list(range(max(0, centre - RASTER_CHANNELS // 2), min(rec.channel_count - 1, centre + RASTER_CHANNELS // 2)))
s0, s1 = int(RASTER_START_S * fs), int((RASTER_START_S + RASTER_LEN_S) * fs)
x = rec.read(channels=chans, start=s0, stop=s1)
t = (np.arange(s0, s1) / fs - RASTER_START_S) * 1000
for j, ch in enumerate(chans):
    ax.plot(t, x[j] + j * CHANNEL_GAP_UV, color="0.35", lw=0.5)
inside = (times >= s0) & (times < s1)
for s, u in zip(times[inside], clusters[inside]):
    ch = peak[u]
    if ch in chans:
        j = chans.index(ch)
        colour = colours[units.index(u) % 10] if u in units else "0.6"
        ax.plot((s / fs - RASTER_START_S) * 1000, j * CHANNEL_GAP_UV, "o", ms=4, color=colour, alpha=0.85)
ax.set_yticks([j * CHANNEL_GAP_UV for j in range(0, len(chans), 4)], [str(chans[j]) for j in range(0, len(chans), 4)])
ax.set_xlabel(f"ms from {RASTER_START_S:g} s")
ax.set_ylabel("channel")
ax.set_title("Spikes over the preprocessed trace (coloured: the units above; grey: other units)", fontsize=10)

fig.suptitle(f"Kilosort4 (dspk) on {rec.channel_count} channels × {rec.samples / fs:.0f} s: "
             f"{len(np.unique(clusters))} units, {len(times)} spikes, {len(good)} good", fontsize=11)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=110)
print(f"wrote {OUT}")
