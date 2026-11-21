---
title: "EMUsort"
description: "Speed per stage, resource usage, unit quality, and comparison against upstream EMUsort 1.1.0."
sidebar:
  order: 2
---

:::note
Measured on 2026-11-21 with `dspk-cli-sorter 0.0.13` in a pre-autotuned GPU container on the
first 200 s of `15-25-33_meps.nwb.zarr` (`HDEMG`, 32 ch @ 24.4 kHz, default `nt = 61`), alongside
upstream Python `emusort 1.1.0` (`kilosort 4.0.18`, `spikeinterface 0.104.7`).
:::

## Processing time

**EMUsort**, HD-EMG (200 s), warm repeat 2:

| Stage | `dspk` wgpu (s) | `dspk` CUDA (s) |
|---|---:|---:|
| Fitting preprocessing | 1.47 | 1.10 |
| Finding clips | 2.43 | 2.21 |
| Learning templates | 0.52 | 0.15 |
| Detecting spikes | 3.00 | 2.01 |
| Clustering spikes | 1.95 | 0.53 |
| Matching templates | 3.23 | 1.97 |
| Clustering matched spikes | 1.57 | 0.67 |
| Merging units | 0.25 | 0.24 |
| **Solver total** | **14.33** | **8.78** |
| **Whole program (e2e)** | **14.67** (13.63× RT) | **9.01** (22.21× RT) |

Upstream Python `EMUsort 1.1.0` on CUDA took **85.06 s** e2e (`70.85 s` solver, `2.35×` real time)
— **9.4× slower than `dspk` CUDA** and **5.8× slower than `dspk` wgpu**.

- **Resources & determinism.** `dspk` peaks at **834 MiB** GPU / **1,007 MiB** RSS on CUDA and
  **631 MiB** GPU / **993 MiB** RSS on wgpu (vs **1,314 MiB** GPU run 1 and **2,861 MiB** RSS in
  upstream EMUsort). Both `dspk` runtimes are bit-exact across repeats; upstream EMUsort produced
  `176 759 / 89` units on run 1 and `177 152 / 102` units on run 2.

## Motor-unit quality & parity checks

There is no ground-truth sorting for `15-25-33_meps`, so units are evaluated by `< 2 ms` inter-spike
interval (ISI) violations, 10 s presence ratio, and spatiotemporal matching (±0.5 ms, 150 µm)
against upstream Run 2 (`102` total units, `34` passing EMUsort quality thresholds):

![EMUsort ISI violations, presence ratio, and unit agreement vs upstream EMUsort](/dsp-kitchen/benchmarks/emusort_hdemg_evaluation.png)

| System | Spikes / Units | Median ISI viol. (< 2 ms) | Units > 1% / > 5% viol. | Presence (median) | Reproducible |
|---|---:|---:|---:|---:|:---:|
| `dspk emusort` (CUDA) | 124 195 / 61 | **0.78%** | 27 / 14 | 1.00 | Bit-exact |
| `dspk emusort` (wgpu) | 127 346 / 78 | **0.62%** | 35 / 11 | 1.00 | Bit-exact |
| Upstream `EMUsort 1.1.0` (CUDA) | 177 152 / 102 | 1.40% | 57 / 15 | 1.00 | Non-exact |

Across the 34 upstream good units, `dspk` recovers **99.5%** median recall with **0.58** (CUDA)
and **0.63** (wgpu) median F1 accuracy (`78%` median precision).

Two earlier design fixes explain the low contamination: dropping duplicate detections at
bit-identical neighbouring centres on wide-pitch HD-EMG grids (before: 94 000 spikes, 53% of a
median unit's intervals under 2 ms) and choosing an appropriate waveform length `nt` (see
[EMUsort settings](/dsp-kitchen/plugins/synapse/reference/settings/emusort/)).
