---
title: "MountainSort 5, SpyKING CIRCUS 2, Tridesclous 2"
description: "Each compared with Kilosort4's saved results on the same recording."
sidebar:
  order: 3
---

:::note
Measured on 2026-11-09 with the release build of `dspk-cli-sorter`, on the setup in the [overview](/dsp-kitchen/plugins/synapse/validation/). Spike counts and agreement come from one run each; the SpyKING CIRCUS 2 and Tridesclous 2 times were measured with the app open, so they may be a little slow. `benchmarks/sorters` has the scripts, and their results are in `benchmarks/sorters/results/`.
:::

## MountainSort 5: agreement with Kilosort4's results

There is no MountainSort 5 reference for this recording: its units are compared with Kilosort4's
saved results as a second sorter on the same data (`mountainsort5_validation.py`, defaults: scheme 2;
the 45 s recording is shorter than the 300 s training stretch, so phase 1 sorts all of it).

| | MountainSort 5 (ours) | Kilosort4 (saved) |
|---|---|---|
| Spikes | 34 942 | 138 666 |
| Units (good by the auto-correlogram) | 167 (134) | 267 (127) |
| Kilosort4 spikes found in time (±0.4 ms) and place (60 µm) | 0.29 | |
| Spike times, ours − Kilosort4 | peak at 0 samples | |
| Kilosort4's good units: found / not split / precision / accuracy (median) | 0.84 / 0.93 / 0.89 / 0.61 | |
| Kilosort4's good units with accuracy ≥ 0.8 / ≥ 0.5 | 0.37 / 0.56 | |

MountainSort 5 finds a quarter of Kilosort4's spikes. Its detection is a single-channel threshold at
5.5 σ after **global** whitening: on this probe the whitened noise has σ = 1.00 (checked), and global
whitening halves the rate of 5.5 σ crossings compared with each channel's own noise. Kilosort4 matches
learned templates and reaches smaller spikes. Where MountainSort 5 does find a unit, it agrees with
Kilosort4's (median precision 0.89, rarely split).

Time: 646 s (10.8 min) for 45 s of 383 channels, against 404 s in October (not investigated). Phase-1 clustering is 471 s: the subdivision tree is lopsided (single linkage peels a few clusters off almost all the spikes at each level, so PCA and isosplit6 run again on nearly every spike, level after level), as in upstream. The per-channel classifiers take 160 s; whitening, detection and classification 15 s.

## SpyKING CIRCUS 2: agreement with Kilosort4's results

As for MountainSort 5 (`spykingcircus2_validation.py`, defaults; spike locations: each unit's main
channel). Not yet in this port: motion correction.

| | SpyKING CIRCUS 2 (ours) | Kilosort4 (saved) |
|---|---|---|
| Spikes | 120 469 | 138 666 |
| Units (good by the auto-correlogram) | 452 (249) | 267 (127) |
| Peaks detected / clustered | 90 067 / 90 067 | |
| Final merges | 4 (456 → 452 units) | |
| Kilosort4 spikes found in time (±0.4 ms) and place (60 µm) | 0.71 | |
| Spike times, ours − Kilosort4 | peak at 0 samples | |
| Kilosort4's good units: found / not split / precision / accuracy (median) | 0.99 / 0.94 / 0.93 / 0.70 | |
| Kilosort4's good units with accuracy ≥ 0.8 / ≥ 0.5 | 0.43 / 0.65 | |

Time: 47 s for 45 s of 383 channels (about real time): preprocessing, prototype, detection and
features 9 s, clustering 11 s, matching 26 s (scalar products on the device a batch of windows at a
time, the pursuits in parallel on the host), final merges and the rest the remainder. More units than Kilosort4: the final merges join only units with at least 100
spikes and little refractory contamination, few of them on a 45 s recording.

## Tridesclous 2: agreement with Kilosort4's results

As for SpyKING CIRCUS 2 (`tridesclous2_validation.py`, defaults).

| | Tridesclous 2 (ours) | Kilosort4 (saved) |
|---|---|---|
| Spikes | 83 110 | 138 666 |
| Units (good by the auto-correlogram) | 562 (371) | 267 (127) |
| Peaks detected / clustered | 57 924 / 57 924 | |
| Final merges | 1 | |
| Kilosort4 spikes found in time (±0.4 ms) and place (60 µm) | 0.59 | |
| Spike times, ours − Kilosort4 | peak at 0 samples | |
| Kilosort4's good units: found / not split / precision / accuracy (median) | 0.98 / 0.90 / 0.98 / 0.80 | |
| Kilosort4's good units with accuracy ≥ 0.8 / ≥ 0.5 | 0.50 / 0.78 | |

Time: 45 s for 45 s of 383 channels (detection 3 s, features 1 s, clustering 2 s, templates 7 s, peeling 28 s with the windows in parallel, final merge 1 s).
Over all 267 Kilosort4 units: found 0.76, precision 0.96, accuracy 0.48 (median).
