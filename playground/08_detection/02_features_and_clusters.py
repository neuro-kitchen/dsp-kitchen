# %% [markdown]
# # 08 · Detection — 2. From spikes to units: snippets, features, clusters
# Each detected spike becomes a **snippet**: a couple of milliseconds around it, on the channels
# near it on the probe. Snippets of one unit look alike (same shape, same channels), so they sit
# together once each is reduced to a few numbers: **principal components** of the snippets. A
# clustering of those points proposes the units. Here the truth is known, so the proposal can be
# checked: each detection is labelled by the true spike it matches.
#
# Two clusterings: **k-means** (you say how many units) and **HDBSCAN** (finds dense groups by
# itself and calls the rest noise, `-1`).

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from dspk.synapse import clustering, detection, extraction, io
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic(channels=8, seconds=4.0, units=3, rate_hz=15.0, noise_uv=10.0)
band = dspk.ops.filter.iir(s.data, s.sample_rate, "bandpass", [300.0, 6000.0])
probe = io.Probe.from_positions([[0, 20 * c] for c in range(s.channels)])
spikes = detection.deduplicate(detection.threshold(band, s.sample_rate), probe, s.sample_rate)

# Snippets on every channel (k = all), rows put back in channel order so features line up
snip = extraction.snippets(band, spikes, probe, s.sample_rate, k=s.channels, pre_ms=0.6, post_ms=1.0)
order = np.argsort(snip["channels"], axis=1)
waves = np.take_along_axis(snip["waveforms"], order[:, :, None], axis=1)  # (spikes, channels, samples)
flat = waves.reshape(waves.shape[0], -1)

# Principal components: dspk's PCA takes (dimensions, observations)
pca = dspk.ops.linalg.PCA.fit(flat.T, 3)
pcs = pca.transform(flat.T).T  # (spikes, 3)
print(f"{flat.shape[0]} snippets of {flat.shape[1]} values → 3 PCs ({pca.explained_variance_ratio.sum():.0%} of the variance)")

# The truth of each detection: the unit of the nearest true spike
offset = int(np.argmin(s.templates[0])) - s.templates.shape[1] // 2
truth_t = s.spike_samples.astype(np.int64) + offset
t = spikes["samples"][snip["kept"]].astype(np.int64)
nearest = np.abs(t[:, None] - truth_t[None, :]).argmin(axis=1)
truth = s.spike_units[nearest]

km = clustering.kmeans(pcs, s.params["units"])
hd = clustering.hdbscan(pcs, min_cluster_size=10)


def agreement(labels: np.ndarray) -> float:
    """Spikes whose cluster's majority unit is their own unit."""
    ok = 0
    for c in np.unique(labels[labels >= 0]):
        members = truth[labels == c]
        ok += np.bincount(members).max()
    return ok / labels.size


print(f"k-means agreement {agreement(km['labels']):.0%} · HDBSCAN agreement {agreement(hd):.0%} ({np.unique(hd[hd >= 0]).size} clusters, {np.mean(hd < 0):.0%} noise)")

# %% Figure: the points coloured by truth and by cluster, and each cluster's mean waveform
p = plating.theme.palette()
fig, axes = plating.figure("From snippets to units: principal components, then clusters", cols=3, size=(13.0, 4.8))
for ax, labels, title in ((axes[0], truth, "true unit"), (axes[1], km["labels"], f"k-means (k = 3): {agreement(km['labels']):.0%} agree")):
    for u in np.unique(labels):
        m = labels == u
        ax.scatter(pcs[m, 0], pcs[m, 1], s=10, color=p.series[int(u) % len(p.series)], label=f"{'unit' if labels is truth else 'cluster'} {u}")
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best", fontsize=8)
for c in np.unique(km["labels"]):
    members = waves[km["labels"] == c]
    peak = int(np.abs(members.mean(axis=0)).max(axis=1).argmax())  # the cluster's own channel
    figures.waveforms(axes[2], members[:, peak], s.sample_rate, color=p.series[int(c) % len(p.series)], label=f"cluster {c} (channel {peak})", show=30)
axes[2].set_title("each cluster on its peak channel (mean solid)")
axes[2].legend(loc="lower right", fontsize=8)
plating.save(fig, FIG / "02_features_and_clusters.png", note="extraction.snippets (all channels) → dspk.ops.linalg.PCA (3) → clustering.kmeans / hdbscan; agreement: spikes whose cluster's majority unit is their own.")
