---
title: "Kilosort4"
description: "Speed per stage, and agreement with Kilosort4's own results."
sidebar:
  order: 1
---

:::note
Measured on 2026-11-09 with the release build of `dspk-cli-sorter`, on the setup in the [overview](/dsp-kitchen/plugins/synapse/validation/). `benchmarks/sorters` has the scripts, and their results are in `benchmarks/sorters/results/`.
:::

## Processing time

**Kilosort4**, Neuropixels, warm (kernels already compiled and tuned):

| Stage | wgpu (s) | CUDA (s) |
|---|---|---|
| Fitting preprocessing | 0.5 | 0.3 |
| Finding clips | 0.2 | 0.2 |
| Learning templates | 0.3 | 0.1 |
| Detecting spikes | 5.3 | 3.0 |
| Clustering spikes | 6.1 | 2.9 |
| Matching templates | 7.5 | 2.9 |
| Clustering matched spikes | 4.5 | 2.3 |
| Merging units | 2.4 | 1.3 |
| **Whole program** | **26.9**: 1.7× real time | **13.0**: 3.4× real time |

The whole program includes starting up and writing the output; the last stage runs to the end of the process. wgpu peaks at 3.2 GB of GPU memory and 1.6 GB of host memory, and two runs give identical spikes and units. A first run on a device also compiles and tunes: CUDA took 39 s the first time here.

**Slower than October.** On wgpu the stages October timed took 19.6 s then and 24.5 s now, mostly in clustering and template matching. The cause hasn't been investigated: the code, the system (a kernel and Mesa update on 2026-11-04) and how stages are timed all changed. Results are unchanged.

## Agreement with Kilosort4's results

Reference: Kilosort4's saved results for the Neuropixels recording (`.datasets/kilosort4/saved_results/`,
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

Earlier stages, checked against the same results (October, except where noted):

| Check | Result |
|---|---|
| Learned `wTEMP` / `wPCA` vs Kilosort4's (best \|cos\| per row) | 0.98–1.00 / 0.94–1.00 |
| Detection with Kilosort4's own learned templates (re-run 2026-11-09) | same match as with ours (0.860 and 0.862): learning is not where they differ |
| Template centres vs `ops['xcup']`, `ops['ycup']` | identical (1532); channel sets differ only among equidistant channels |
| Spike times | difference ours − Kilosort4 peaks at 0 samples (troughs, as Kilosort4) |
| Spike positions (a well-isolated unit) | median 8 µm apart |
