# %% [markdown]
# # 09 · Sorting — 1. A real probe: every spike, where it came from, and how the probe moved
# Kilosort4's test recording: 45 s of a Neuropixels 1.0 probe (384 sites over 3.8 mm, plus a
# sync channel), 30 kHz, int16. First run `00_fetch_kilosort_data.py`.
#
# 1. **Detect** over the whole file, streaming on the device (`detect_recording`: band-pass, common
#    median reference, threshold, merging across sites) — what `dspk detect` does.
# 2. **Locate** each spike: snippets on its nearest sites, then the centre of mass of their
#    peak-to-peak amplitudes (µm on the probe). Read in windows, so memory stays small.
# 3. **Drift**: spikes' depths over time show the probe moving in the brain (bands of activity
#    bend); `spatial.drift` estimates that motion — what sorters correct before clustering.

# %%
from pathlib import Path

import numpy as np

import dspk
import plating
from dspk.synapse import detection, extraction, io, spatial

FIG = Path(__file__).parent / "figures"
DATA = Path(__file__).resolve().parents[1] / "data" / "kilosort4" / "ZFM-02370_mini.imec0.ap.short.bin"
if not DATA.exists():
    raise SystemExit(f"{DATA} is missing: run playground/09_sorting/00_fetch_kilosort_data.py first")

rec = dspk.io.open(DATA)
probe = io.Probe.preset("neuropixels1")  # 384 sites; the file's channel 385 is the sync line
fs, total = rec.sample_rate, rec.samples
print(rec)

# 1. Detection over the whole recording (streamed in batches on the device)
sorting, summary = detection.detect_recording(rec, probe, car=True)
spikes = sorting.spikes()
print(f"{summary['spikes']} spikes after merging ({summary['raw_crossings']} threshold crossings)")

# 2. Locate: windows of WINDOW_S, each band-passed and median-referenced, snippets of its spikes
WINDOW_S, MARGIN_S = 5.0, 0.05
sites = list(range(len(probe)))
positions = np.full((spikes["samples"].size, 3), np.nan, dtype=np.float32)
step, margin = int(WINDOW_S * fs), int(MARGIN_S * fs)
for a in range(0, total, step):
    lo, hi = max(0, a - margin), min(total, a + step + margin)
    x = dspk.ops.spatial.cmr(dspk.ops.filter.iir(rec.read(channels=sites, start=lo, stop=hi), fs, "bandpass", [300.0, 6000.0]))
    inside = np.flatnonzero((spikes["samples"] >= a) & (spikes["samples"] < min(a + step, total)))
    local = {"samples": spikes["samples"][inside] - lo, "channels": spikes["units"][inside].astype(np.uint32)}
    snip = extraction.snippets(x, local, probe, fs, k=8, pre_ms=0.5, post_ms=1.0)
    positions[inside[snip["kept"]]] = spatial.localize(snip, probe, method="center_of_mass")
ok = ~np.isnan(positions[:, 1])
print(f"located {ok.mean():.0%} of the spikes")

# 3. Drift from the located spikes (rigid: one displacement for the whole probe per time bin)
t, depth, amp = spikes["samples"][ok], positions[ok, 1], np.abs(spikes["amplitudes"][ok])
drift = spatial.drift(t, depth, amp, total, fs, depth_min_um=float(depth.min()), depth_max_um=float(depth.max()))
print(f"drift over the recording: {np.ptp(drift['drift_um']):.1f} µm peak to peak")

# %% Figure: the classic drift map (each spike at its time and depth, darker = larger), the drift
p = plating.theme.palette()
fig, (left, right) = plating.figure("A Neuropixels probe: every spike at its depth, and how the probe moved", cols=2, size=(13.0, 6.5))
order = np.argsort(amp)
left.scatter(t[order] / fs, depth[order], s=0.6, c=np.log10(amp[order]), cmap="Greys", vmin=1.6, vmax=2.6, rasterized=True)
left.set_xlabel("time (s)")
left.set_ylabel("depth on the probe (µm)")
left.set_title(f"{t.size:,} spikes (centre of mass of their sites)")
right.plot(drift["times_s"], drift["drift_um"] - drift["drift_um"][0], color=p.series[0], marker="o", markersize=3)
plating.theme.reference_line(right, 0.0, "no motion")
right.set_xlabel("time (s)")
right.set_ylabel("displacement (µm)")
right.set_ylim(-10, 10)  # on the scale of a site pitch (20 µm): this probe barely moved
right.set_title(f"rigid drift estimate: {np.ptp(drift['drift_um']):.1f} µm, the probe barely moved")
plating.save(fig, FIG / "01_detect_and_locate.png", note="detection.detect_recording (300–6000 Hz, CMR) → extraction.snippets (8 sites) → spatial.localize (center of mass) → spatial.drift (rigid).")
