---
title: "Kilosort4"
description: "Speed per stage, and agreement with Kilosort4's own results."
sidebar:
  order: 1
---

:::note
Measured in October 2026 on the setup in the [overview](/dsp-kitchen/plugins/synapse/validation/). The scripts in `benchmarks/sorters` still use the previous Python API and don't run on the current package, so these results haven't been re-run since.
:::

## Processing time

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

Earlier stages, checked against the same results:

| Check | Result |
|---|---|
| Learned `wTEMP` / `wPCA` vs Kilosort4's (best \|cos\| per row) | 0.98–1.00 / 0.94–1.00 |
| Detection with Kilosort4's own learned templates | same match as with ours: learning is not where they differ |
| Template centres vs `ops['xcup']`, `ops['ycup']` | identical (1532); channel sets differ only among equidistant channels |
| Spike times | difference ours − Kilosort4 peaks at 0 samples (troughs, as Kilosort4) |
| Spike positions (a well-isolated unit) | median 8 µm apart |
