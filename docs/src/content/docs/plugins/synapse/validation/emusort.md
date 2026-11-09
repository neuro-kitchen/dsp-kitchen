---
title: "EMUsort"
description: "Speed per stage, and checks that need no reference."
sidebar:
  order: 2
---

:::note
Measured in October 2026 on the setup in the [overview](/dsp-kitchen/plugins/synapse/validation/). The scripts in `benchmarks/sorters` still use the previous Python API and don't run on the current package, so these results haven't been re-run since.
:::

## Processing time

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


## Checks without ground truth

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
[EMUsort settings](/dsp-kitchen/plugins/synapse/reference/settings/emusort/)).
