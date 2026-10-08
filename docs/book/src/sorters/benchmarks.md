# Benchmarks

Processing times of our sorters on the two test recordings, and how their output compares with
Kilosort4's own results where those exist. Kilosort4's and EMUsort's processing times are not given:
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
| Spikes | 132 241 | 138 666 |
| Units | 293 | 267 |
| Kilosort4 spikes found (time and place) | 89.3% | |

Per Kilosort4 unit (median, 25th percentile in brackets):

| | All 267 units | 127 good units |
|---|---|---|
| Spikes found | 0.95 (0.85) | 1.00 (0.97) |
| Of those, in one unit of ours (not split) | 0.91 (0.68) | 0.97 (0.87) |
| That unit's spikes that are this unit's (precision) | 0.85 (0.57) | 0.94 (0.56) |
| Accuracy `found / (n_ks + n_ours − found)` | 0.62 (0.39) | 0.82 (0.49) |
| Units with accuracy ≥ 0.8 / ≥ 0.5 | 30% / 64% | 53% / 73% |

The export round trip passes: the Phy folder and the `.sorting.zarr` read back the same spikes and
units, and the Phy folder has every file Kilosort4 writes (templates included) except Kilosort's
internal ones. Per channel, 88.8% of Kilosort4's spikes have one of ours within ±0.4 ms and 40 µm,
93.1% of ours have one of Kilosort4's, and the time differences peak at 0. A matched spike is placed
at its template's position, so ours sit on fewer channels (189) than Kilosort4's per-spike positions
(350).

How each stage moved the good units' median accuracy: universal-template detection with one unit per
universal template is not comparable (6 "units"); with the first clustering, 0.46; with learned-
template matching and the clustering of its spikes, 0.82. The remaining gap is mostly units whose
precision is low (a quarter of the good units below 0.55: two neurons in one unit), which the
refractory criteria and the global merges (not yet implemented) address.

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

## Running the benchmarks

```bash
uv run python playground/benchmarks/kilosort4_validation.py     # export round trip, per-channel spikes, per-unit agreement
uv run python playground/benchmarks/kilosort4_experiments.py    # learned vs Kilosort4's templates
uv run python playground/benchmarks/kilosort4_benchmark.py      # speed per stage and runtime
uv run python playground/benchmarks/emusort_checks.py 200       # EMUsort on a 200 s segment
```

`playground/benchmarks/ks4_reference.py` holds the shared comparison (Kilosort4's saved results, the
per-channel and per-unit matching).
