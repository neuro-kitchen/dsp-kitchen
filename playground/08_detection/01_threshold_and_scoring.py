# %% [markdown]
# # 08 · Detection — 1. Finding spikes, and scoring what was found
# A spike is a brief deflection well beyond the noise. The simplest detector marks every local
# minimum below `threshold · σ`, with σ estimated per channel from the median absolute deviation
# (robust to the spikes themselves). Two others look at the shape: **NEO** (Teager–Kaiser energy,
# large where the signal is both big and fast) and a **matched filter** (correlation with a spike
# shape). One spike shows on neighbouring channels, so detections within a radius on the probe
# and a fraction of a millisecond are merged (`deduplicate`).
#
# The synthetic recording knows when each unit fired, so every detector can be scored: a true
# spike is *found* when a detection lands within 0.5 ms of it; a detection is *false* when no
# true spike is near.

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from dspk.synapse import detection, io
from plating import figures, signals

FIG = Path(__file__).parent / "figures"
s = signals.synthetic(channels=8, seconds=4.0, units=3, rate_hz=15.0, noise_uv=10.0)
band = dspk.ops.filter.iir(s.data, s.sample_rate, "bandpass", [300.0, 6000.0])
probe = io.Probe.from_positions([[0, 20 * c] for c in range(s.channels)])  # a linear probe, 20 µm pitch

# Where each true spike's trough is (the synthetic places the waveform's centre at spike_samples)
offset = int(np.argmin(s.templates[0])) - s.templates.shape[1] // 2
truth = np.sort(s.spike_samples.astype(np.int64) + offset)
TOL = int(0.5e-3 * s.sample_rate)


def nearest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance (samples) from each of ``a`` to the nearest of ``b`` (sorted)."""
    i = np.searchsorted(b, a)
    left, right = b[np.clip(i - 1, 0, b.size - 1)], b[np.clip(i, 0, b.size - 1)]
    return np.minimum(np.abs(a - left), np.abs(a - right))


def score(found: np.ndarray) -> tuple[float, float]:
    """Recall (true spikes found) and precision (detections that are true)."""
    found = np.sort(found.astype(np.int64))
    if found.size == 0:
        return 0.0, 0.0
    return float(np.mean(nearest(truth, found) <= TOL)), float(np.mean(nearest(found, truth) <= TOL))


results = {}
for name, found in {
    "threshold": detection.threshold(band, s.sample_rate),
    "NEO": detection.neo(band, s.sample_rate),
    "matched filter": detection.matched_filter(band, s.sample_rate),
}.items():
    merged = detection.deduplicate(found, probe, s.sample_rate)
    results[name] = (found, merged, *score(merged["samples"]))
    print(f"{name:15s} {found['samples'].size:5d} crossings → {merged['samples'].size:4d} spikes · recall {results[name][2]:.0%} · precision {results[name][3]:.0%}")

# %% Figure: the threshold on the loudest channel, and every detector's score
p = plating.theme.palette()
ch = s.loudest_channel
sigma = np.median(np.abs(band[ch])) / 0.6745
fig, (top, bottom) = plating.figure("Detection: threshold crossings, then one spike per event", rows=2, size=(11.0, 8.0), height_ratios=[2, 1])
window = (0.66, 0.76)
t = np.arange(band.shape[1]) / s.sample_rate
w = (t >= window[0]) & (t < window[1])
top.plot(t[w], band[ch, w], color=p.series[0], linewidth=0.8, label=f"channel {ch}, 300–6000 Hz")
plating.theme.reference_line(top, -detection.DEFAULT_THRESHOLD * sigma, f"−{detection.DEFAULT_THRESHOLD:g} σ")
found, merged = results["threshold"][:2]
on = (merged["channels"] == ch) & (merged["samples"] / s.sample_rate >= window[0]) & (merged["samples"] / s.sample_rate < window[1])
top.plot(merged["samples"][on] / s.sample_rate, band[ch, merged["samples"][on].astype(int)], "v", color=p.series[1], label="detected (after merging)")
tt = truth[(truth / s.sample_rate >= window[0]) & (truth / s.sample_rate < window[1])]
top.plot(tt / s.sample_rate, np.full(tt.size, band[ch, w].max() * 1.1), "|", color=p.muted, markersize=10, label="true spikes (every unit, any channel)")
top.set_xlabel("time (s)")
top.set_ylabel("µV")
top.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncols=3, frameon=False)
names = list(results)
x = np.arange(len(names))
bottom.bar(x - 0.18, [results[n][2] * 100 for n in names], 0.36, color=p.series[0], label="recall (true spikes found)")
bottom.bar(x + 0.18, [results[n][3] * 100 for n in names], 0.36, color=p.series[2], label="precision (detections that are true)")
bottom.set_xticks(x, names)
bottom.set_ylim(0, 135)
bottom.set_ylabel("%")
bottom.legend(loc="upper right", ncols=2)
plating.save(fig, FIG / "01_threshold_and_scoring.png", note="dspk.synapse.detection: threshold / neo / matched_filter, then deduplicate (radius on the probe, 0.5 ms); scored against the synthetic ground truth (±0.5 ms).")
