---
title: "EMUsort"
description: "Speed per stage, and checks that need no reference."
sidebar:
  order: 2
---

:::note
Measured on 2026-11-09 with the release build of `dspk-cli-sorter`, on the setup in the [overview](/dsp-kitchen/plugins/synapse/validation/). `benchmarks/sorters` has the scripts, and their results are in `benchmarks/sorters/results/`.
:::

## Processing time

**EMUsort**, HD-EMG, 200 s, wgpu, warm (default `nt = 61`):

| Stage | Time (s) |
|---|---|
| Fitting preprocessing | 1.84 |
| Finding clips | 2.09 |
| Learning templates | 0.96 |
| Detecting spikes | 2.82 |
| Clustering spikes | 3.67 |
| Matching templates | 3.45 |
| Clustering matched spikes | 4.13 |
| Merging units | 0.61 |
| **Whole program** | **19.6**: 10.2× real time |

Peaks at 0.55 GB of GPU memory and 0.65 GB of host memory. Two runs of the same input give identical spikes and units.

**Slower than October**, 15.3 s then and 19.0 s now on the stages October timed. As for [Kilosort4](/dsp-kitchen/plugins/synapse/validation/kilosort4/), the cause hasn't been investigated.


## Checks without ground truth

There is no reference sorting for the HD-EMG recording, so the checks are those a reference does not
need (first 200 s, default `nt = 61`):

| | |
|---|---|
| Spikes / units | 126 153 / 69 |
| Units with > 1% / > 5% of inter-spike intervals < 2 ms | 33 / 9 |
| Presence over the segment (median) | 1.00 |

October's run also compared `nt = 121` (103 362 spikes, 51 units, 0.8% of template energy in the outer 10% of the window against 4.1% at `nt = 61`). That comparison hasn't been repeated.

Two fixes came out of these checks: duplicate detections at bit-identical neighbouring centres (before:
94 000 spikes, 53% of a median unit's intervals under 2 ms) and the template length (see
[EMUsort settings](/dsp-kitchen/plugins/synapse/reference/settings/emusort/)).
