---
title: "Kilosort4"
description: "Speed per stage, resource usage, and agreement against Kilosort4 4.1.3 and saved results."
sidebar:
  order: 1
---

:::note
Measured on 2026-11-21 with `dspk-cli-sorter 0.0.13` in a pre-autotuned GPU container on
`ZFM-02370_mini.nwb.zarr` (45 s, 383 ch @ 30 kHz), alongside direct `kilosort 4.1.3` and
`spikeinterface 0.106.0` on the [validation setup](/dsp-kitchen/plugins/synapse/validation/). Raw
records are in `benchmarks/sorters/results/`.
:::

## Processing time

**Kilosort4**, Neuropixels (45 s), warm repeat 2 (kernels already compiled and autotuned):

| Stage | `dspk` wgpu (s) | `dspk` CUDA (s) | Upstream KS4 `4.1.3` (CUDA, s) |
|---|---:|---:|---:|
| Fitting preprocessing | 0.41 | 0.25 | 0.55 |
| Finding clips | 0.35 | 0.38 | — |
| Learning templates | 0.17 | 0.09 | — |
| Computing drift | 5.44 | 4.69 | 30.12 |
| Detecting spikes | 5.75 | 4.26 | 24.69 |
| Clustering spikes | 3.18 | 0.96 | 16.82 |
| Matching templates | 5.31 | 3.79 | 8.04 |
| Clustering matched spikes | 2.08 | 0.62 | 16.95 |
| Merging units | 0.46 | 0.49 | 0.41 |
| **Solver total** | **23.15** | **15.50** | **106.07** |
| **Whole program (e2e)** | **23.56** (1.91× RT) | **15.76** (2.86× RT) | **112.42** (0.40× RT) |

SpikeInterface `0.106.0` wrapping Kilosort4 `4.1.3` took **114.79 s** e2e (`107.48 s` solver). On
this recording, `dspk` is **7.13× faster on CUDA** and **4.77× faster on wgpu** than upstream
PyTorch CUDA Kilosort4.

- **Resources & determinism.** CUDA peaks at **3,754 MiB** GPU memory and **1,313 MiB** RSS; wgpu
  peaks at **3,232 MiB** GPU memory and **1,438 MiB** RSS — compared with **5,432 MiB** peak GPU
  *(run 1)* and **3,752 MiB** RSS in upstream Kilosort4. Repeat runs of `dspk` produce bit-exact
  spikes, clusters, amplitudes, locations, and templates on both backends.
- **`wgpu` optimization impact.** Shared-memory staging in `centre_response_kernel`, amplitude-only
  `STAGE_DRIFT` detection, and precomputed per-batch kriging matrices reduced `wgpu` end-to-end
  time from `250.53 s` to `23.56 s` (`Computing drift` `119.53 s → 5.44 s`, `Detecting spikes`
  `118.61 s → 5.75 s`) with zero byte differences ([why and how](/dsp-kitchen/gpu/detection/)).

![Stage-by-stage latency comparison for Kilosort4 and EMUsort](/dsp-kitchen/benchmarks/sorter_stage_latency.png)

## Agreement with Kilosort4's results

**How spikes are compared.** Unit ids differ between sorters, so an upstream spike counts as found
when one of ours lies within **±0.4 ms** and **40 µm** of its position (`spike_positions.npy`). Each
upstream unit is paired with the unit of ours holding most of its found spikes.

![Kilosort4 unit parity, precision-recall, and temporal lag](/dsp-kitchen/benchmarks/ks4_parity_accuracy.png)

| Comparison (±0.4 ms, 40 µm) | Spikes / Units (good) | Total spikes found | Good units: found / not split / precision / F1 (median) | Template cosine (median) |
|---|---|---:|---:|---:|
| `dspk (cuda)` vs live KS4 `4.1.3` | 128 325 / 283 (149) | 85.3% | 1.00 / 0.98 / 0.93 / **0.86** | 0.946 |
| `dspk (wgpu)` vs live KS4 `4.1.3` | 128 416 / 278 (145) | 85.9% | 1.00 / 0.97 / 0.95 / **0.86** | 0.946 |
| `SpikeInterface 0.106` vs live KS4 `4.1.3` | 137 374 / 263 (126) | 100.0% | 1.00 / 1.00 / 1.00 / 1.00 | 1.000 |
| `dspk (cuda)` vs `saved_results/` | 128 325 / 283 (149) | 85.2% | 1.00 / 0.98 / 0.91 / 0.80 | — |
| `dspk (wgpu)` vs `saved_results/` | 128 416 / 278 (145) | 86.1% | 1.00 / 0.98 / 0.94 / 0.82 | — |
| Live KS4 `4.1.3` vs `saved_results/` | 137 374 / 263 (126) | 95.5% | 1.00 / 0.98 / 0.95 / 0.90 | — |

On good units against live Kilosort4 `4.1.3`, **54.8%** (`cuda`) and **59.5%** (`wgpu`) reach
accuracy ≥ 0.8 (**75.4%** and **74.6%** ≥ 0.5), with peak spike-time lag at **0 samples**.

Earlier stage-by-stage checks against `saved_results/`:

| Check | Result |
|---|---|
| Learned `wTEMP` / `wPCA` vs Kilosort4's (best \|cos\| per row) | 0.98–1.00 / 0.94–1.00 |
| Template centres vs `ops['xcup']`, `ops['ycup']` | identical (1532); channel sets differ only among equidistant channels |
| Spike times | difference ours − Kilosort4 peaks at 0 samples (troughs, as Kilosort4) |
| Spike positions (a well-isolated unit) | median 8 µm apart |
