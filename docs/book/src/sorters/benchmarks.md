# Benchmarks

Processing times of our sorters on the two test recordings, and how their output compares with
Kilosort4's own results where those exist (MountainSort 5 included, as a second sorter). Kilosort4's and EMUsort's processing times are not given:
we have no run of the originals on this machine, and a time recorded on other hardware would not be
comparable.

## Setup

| | |
|---|---|
| Device | NVIDIA GeForce RTX 2070 with Max-Q Design (8 GB), driver 615.71 |
| Host | Intel Core i7-9750H, 12 threads |
| Runtime | wgpu (Vulkan), `f32`, `reproducible = true` |
| Build | `maturin develop --release`, Python 3.13 |

Times are wall-clock from Python, split by the stages of the progress report (time between two
reports counts for the stage reporting; the learned templates, between clustering and matching,
count for matching). **Cold** is the first run of a process (kernels compile); **warm** the second.

| Recording | Channels | Duration | Rate | Data |
|---|---|---|---|---|
| Neuropixels 1.0, `ZFM-02370_mini.imec0.ap.short.bin` | 383 | 45 s | 30 kHz | `data/kilosort4/` |
| HD-EMG 4 × 8 grid, `15-25-33_meps.nwb.zarr` (HDEMG series), first 200 s | 32 | 200 s | 24.4 kHz | `data/nwb/` |

## Processing times

**Kilosort4**, Neuropixels (warm; cold in brackets):

| Stage | Time (s) |
|---|---|
| Fitting preprocessing | 0.04 (0.40) |
| Finding clips | 0.31 (0.49) |
| Learning templates | 0.22 (0.27) |
| Detecting spikes | 5.99 (5.05) |
| Clustering spikes | 4.30 (5.14) |
| Learned templates + matching | 5.15 (7.72) |
| Clustering matched spikes | 3.61 (4.32) |
| **Total** | **19.6 (23.4)**: 2.3× real time (1.9× cold) |

**EMUsort**, HD-EMG, 200 s (warm; cold in brackets):

| Stage | `nt = 61` | `nt = 121` |
|---|---|---|
| Fitting preprocessing (incl. channel delays) | 1.12 (1.44) | 1.21 (1.42) |
| Finding clips | 2.27 (2.64) | 2.02 (2.62) |
| Learning templates (incl. HDBSCAN) | 0.60 (0.72) | 0.62 (0.99) |
| Detecting spikes | 2.39 (2.53) | 2.48 (2.60) |
| Clustering spikes | 2.20 (2.84) | 1.60 (2.12) |
| Learned templates + matching | 3.32 (3.50) | 4.19 (4.19) |
| Clustering matched spikes | 3.42 (3.53) | 2.65 (2.99) |
| **Total** | **15.3 (17.2)**: 13.0× real time | **14.8 (16.9)**: 13.5× real time |

Two runs of the same input give identical spikes and units on both recordings.

## Kilosort4: agreement with Kilosort4's results

Reference: Kilosort4's saved results for the Neuropixels recording (`data/kilosort4/saved_results/`,
a Phy folder: 138 666 spikes, 267 units of which 127 labelled *good*). Its settings (`ops.npy`) equal
our defaults. Kilosort4 ran drift correction; its correction on this recording is ±0.5 µm.

**How spikes are compared.** Unit ids differ between sorters, so a Kilosort4 spike counts as found
when one of ours lies within ±0.4 ms **and** within 40 µm of its position (`spike_positions.npy`).
Time alone is not enough at this spike density: some spike of ours falls within ±0.4 ms of any
instant 80% of the time. Each Kilosort4 unit is paired with the unit of ours holding most of its
found spikes.

| | Ours | Kilosort4 |
|---|---|---|
| Spikes | 130 551 | 138 666 |
| Units (good by the refractory labels) | 291 (152) | 267 (127) |
| Kilosort4 spikes found (time and place) | 86.6% | |

Per Kilosort4 unit (median, 25th percentile in brackets):

| | All 267 units | 127 good units |
|---|---|---|
| Spikes found | 0.95 (0.82) | 1.00 (0.97) |
| Of those, in one unit of ours (not split) | 0.91 (0.73) | 0.98 (0.93) |
| That unit's spikes that are this unit's (precision) | 0.82 (0.53) | 0.95 (0.52) |
| Accuracy `found / (n_ks + n_ours − found)` | 0.57 (0.39) | 0.82 (0.45) |
| Units with accuracy ≥ 0.8 / ≥ 0.5 | 30% / 60% | 52% / 72% |

The export round trip passes: the Phy folder and the `.sorting.zarr` read back the same spikes and
units, and the Phy folder has every file Kilosort4 writes (templates included) except Kilosort's
internal ones. Per channel, 86.2% of Kilosort4's spikes have one of ours within ±0.4 ms and 40 µm,
91.5% of ours have one of Kilosort4's, and the time differences peak at 0. A matched spike is placed
at its template's position, so ours sit on fewer channels (197) than Kilosort4's per-spike positions
(350).

How each stage moved the good units' median accuracy: universal-template detection with one unit per
universal template is not comparable (6 "units"); with the first clustering, 0.46; with learned-
template matching and the clustering of its spikes, 0.82. The remaining gap is mostly units whose
precision is low (a quarter of the good units below 0.55: two neurons in one unit), which the
refractory criteria and the global merges were meant to address; with both in place (numbers above,
rerun 2026-10-09) the good units' median precision is 0.95, its 25th percentile still 0.52.

Earlier stages, checked against the same results:

| Check | Result |
|---|---|
| Learned `wTEMP` / `wPCA` vs Kilosort4's (best \|cos\| per row) | 0.98–1.00 / 0.94–1.00 |
| Detection with Kilosort4's own learned templates | same match as with ours: learning is not where they differ |
| Template centres vs `ops['xcup']`, `ops['ycup']` | identical (1532); channel sets differ only among equidistant channels |
| Spike times | difference ours − Kilosort4 peaks at 0 samples (troughs, as Kilosort4) |
| Spike positions (a well-isolated unit) | median 8 µm apart |

## EMUsort: checks without ground truth

There is no reference sorting for the HD-EMG recording, so the checks are those a reference does not
need (first 200 s, `nt = 61` / `121`):

| | `nt = 61` | `nt = 121` |
|---|---|---|
| Spikes / units | 125 603 / 62 | 103 362 / 51 |
| Units with > 1% / > 5% of inter-spike intervals < 2 ms | 38 / 13 | 30 / 9 |
| Template energy in the outer 10% of the window (truncation) | 4.1% | 0.8% |
| Presence over the segment (median) | 1.00 | 1.00 |

Two fixes came out of these checks: duplicate detections at bit-identical neighbouring centres (before:
94 000 spikes, 53% of a median unit's intervals under 2 ms) and the template length (see
[EMUsort parameters](emusort/parameters.md)).

## MountainSort 5: agreement with Kilosort4's results

There is no MountainSort 5 reference for this recording: its units are compared with Kilosort4's
saved results as a second sorter on the same data (`mountainsort5_validation.py`, defaults: scheme 2;
the 45 s recording is shorter than the 300 s training stretch, so phase 1 sorts all of it). Build:
`--profile validate` (optimised, not `--release`).

| | MountainSort 5 (ours) | Kilosort4 (saved) |
|---|---|---|
| Spikes | 35 121 | 138 666 |
| Units (good by the auto-correlogram) | 165 (138) | 267 (127) |
| Kilosort4 spikes found in time (±0.4 ms) and place (60 µm) | 0.29 | |
| Spike times, ours − Kilosort4 | peak at 0 samples | |
| Kilosort4's good units: found / not split / precision / accuracy (median) | 0.83 / 0.95 / 0.91 / 0.59 | |
| Kilosort4's good units with accuracy ≥ 0.8 / ≥ 0.5 | 0.36 / 0.58 | |

MountainSort 5 finds a quarter of Kilosort4's spikes. Its detection is a single-channel threshold at
5.5 σ after **global** whitening: on this probe the whitened noise has σ = 1.00 (checked), and global
whitening halves the rate of 5.5 σ crossings compared with each channel's own noise. Kilosort4 matches
learned templates and reaches smaller spikes. Where MountainSort 5 does find a unit, it agrees with
Kilosort4's (median precision 0.91, rarely split).

Time: 404 s (6.7 min) for 45 s of 383 channels. Phase-1 clustering is 297 s: the subdivision tree is
lopsided (single linkage peels a few clusters off almost all 39 649 spikes at each level, so PCA and
isosplit6 run again on nearly every spike, level after level), as in upstream; the per-channel
classifiers take about 90 s; detection, training snippets and classification 17 s.

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

Time: 44 s for 45 s of 383 channels (about real time): preprocessing, prototype, detection and
features 4 s, clustering 6 s, matching 20 s (scalar products on the device a batch of windows at a
time, the pursuits in parallel on the host; 118 s with one window at a time), final merges and the
rest the remainder. More units than Kilosort4: the final merges join only units with at least 100
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

Time: 41 s for 45 s of 383 channels (detection 2 s, features 2 s, clustering 3 s, templates 1 s,
peeling 19 s with the windows in parallel, final merge 1 s; the rest is loading and setup).
Over all 267 Kilosort4 units: found 0.76, precision 0.96, accuracy 0.48 (median).

## Summary on the Kilosort4 recording

| Sorter | Time (s) | Spikes | Units | KS4 good: found / precision / accuracy (median) | KS4 good with accuracy ≥ 0.8 |
|---|---|---|---|---|---|
| Kilosort4 (ours)¹ | — | 130 551 | 291 | 1.00 / 0.95 / 0.82 | 0.52 |
| MountainSort 5 | 404 | 35 121 | 165 | 0.83 / 0.91 / 0.59 | 0.36 |
| SpyKING CIRCUS 2 | 44 | 120 469 | 452 | 0.99 / 0.93 / 0.70 | 0.43 |
| Tridesclous 2 | 41 | 83 110 | 562 | 0.98 / 0.98 / 0.80 | 0.50 |

Kilosort4 is the reference here, so these measure agreement with it, not accuracy against ground
truth. SpyKING CIRCUS 2 and Tridesclous 2 split units more than Kilosort4 (452 and 562 units).

¹ Matched within 40 µm (the other rows: 60 µm), from the Kilosort4 section above.

## Running the benchmarks

```bash
uv run python playground/benchmarks/kilosort4_validation.py     # export round trip, per-channel spikes, per-unit agreement
uv run python playground/benchmarks/kilosort4_experiments.py    # learned vs Kilosort4's templates
uv run python playground/benchmarks/kilosort4_benchmark.py      # speed per stage and runtime
uv run python playground/benchmarks/emusort_checks.py 200       # EMUsort on a 200 s segment
uv run python playground/benchmarks/mountainsort5_validation.py --rerun   # MountainSort 5 vs Kilosort4 (~7 min)
uv run python playground/benchmarks/spykingcircus2_validation.py --rerun  # SpyKING CIRCUS 2 vs Kilosort4 (~1 min)
uv run python playground/benchmarks/tridesclous2_validation.py --rerun    # Tridesclous 2 vs Kilosort4 (~1 min)
```

`playground/benchmarks/ks4_reference.py` holds the shared comparison (Kilosort4's saved results, the
per-channel and per-unit matching).
