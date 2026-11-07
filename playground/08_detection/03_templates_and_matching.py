# %% [markdown]
# # 08 · Detection — 3. Templates, and finding every spike again by matching
# A cluster's mean snippet is its unit's **template**. With templates in hand, a recording can be
# searched for them directly: **matching pursuit** fits, pass after pass, the template (and its
# amplitude) that explains the most of what is left, and subtracts it. Spikes that overlap in
# time, which a threshold sees as one, come apart over the passes. Template **similarity**
# (cosine at the best lag) shows which units could be one split in two.

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from dspk.synapse import clustering, detection, extraction, io, matching
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic(channels=8, seconds=4.0, units=3, rate_hz=15.0, noise_uv=10.0)
band = dspk.ops.filter.iir(s.data, s.sample_rate, "bandpass", [300.0, 6000.0])
probe = io.Probe.from_positions([[0, 20 * c] for c in range(s.channels)])

# Chapter 2 in four lines: detect, snippets on every channel, PCs, k-means
spikes = detection.deduplicate(detection.threshold(band, s.sample_rate), probe, s.sample_rate)
snip = extraction.snippets(band, spikes, probe, s.sample_rate, k=s.channels, pre_ms=0.6, post_ms=1.0)
waves = np.take_along_axis(snip["waveforms"], np.argsort(snip["channels"], axis=1)[:, :, None], axis=1)
flat = waves.reshape(waves.shape[0], -1)
labels = clustering.kmeans(dspk.ops.linalg.PCA.fit(flat.T, 3).transform(flat.T).T, 3)["labels"]

# Templates: each cluster's mean snippet, (units, channels, samples)
templates = np.stack([waves[labels == c].mean(axis=0) for c in range(3)]).astype(np.float32)
similarity = matching.template_similarity(templates)
print("template similarity:\n", np.round(similarity, 2))
print("merge suggestions (≥ 0.9):", matching.suggest_merges(templates))

found = matching.matching_pursuit(band, templates, min_explained_energy=20_000.0)
print(f"matching pursuit: {found['samples'].size} spikes (threshold + merging found {spikes['samples'].size}; truth {s.spike_samples.size})")

# Scored against the truth (±0.5 ms), unit by unit (each template to the true unit it holds most of)
offset = int(np.argmin(s.templates[0])) - s.templates.shape[1] // 2
truth_t = s.spike_samples.astype(np.int64) + offset
tol = int(0.5e-3 * s.sample_rate)
for c in range(3):
    mine = np.sort(found["samples"][found["units"] == c].astype(np.int64))
    near = np.abs(truth_t[:, None] - mine[None, :]).min(axis=1) <= tol if mine.size else np.zeros(truth_t.size, bool)
    unit = np.bincount(s.spike_units[near], minlength=3).argmax() if near.any() else -1
    recall = np.mean(near[s.spike_units == unit]) if unit >= 0 else 0.0
    print(f"template {c} → true unit {unit}: {mine.size} matches, {recall:.0%} of that unit's spikes")

# %% Figure: templates, their similarity, and what matching found over time
p = plating.theme.palette()
fig, axes = plating.figure("Templates, and the recording searched for them", cols=3, size=(13.0, 4.8))
gap = np.abs(templates).max() * 1.2
t_ms = (np.arange(templates.shape[2]) - round(0.6e-3 * s.sample_rate)) / s.sample_rate * 1e3
for c in range(3):
    for ch in range(s.channels):
        axes[0].plot(t_ms + c * (t_ms[-1] - t_ms[0] + 0.4), templates[c, ch] - ch * gap, color=p.series[c], linewidth=1.2)
axes[0].set_yticks([-ch * gap for ch in range(s.channels)], [f"ch {ch}" for ch in range(s.channels)])
axes[0].set_xticks([])
axes[0].set_title("templates 0 · 1 · 2, every channel")
figures.matrix(axes[1], similarity, diverging=False, labels=[f"T{c}" for c in range(3)], colorbar_label="cosine at best lag")
axes[1].set_title("template similarity")
figures.raster(axes[2], found["samples"], found["units"], s.sample_rate, unit_names=[f"template {c}" for c in range(3)])
axes[2].set_xlim(0, 1.0)
axes[2].set_title("matches over the first second")
plating.save(fig, FIG / "03_templates_and_matching.png", note="matching.template_similarity / suggest_merges; matching.matching_pursuit (min_explained_energy 20 000 µV²) over the 300–6000 Hz band.")
