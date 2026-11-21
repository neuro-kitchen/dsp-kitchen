---
title: Validation
description: "Each sorter against Kilosort4's saved results, live Kilosort4 4.1.3, SpikeInterface 0.106, and EMUsort 1.1.0."
sidebar:
  label: Overview
  order: 0
---

:::note
Kilosort4, SpikeInterface KS4, and EMUsort were measured on 2026-11-21 with `dspk 0.0.13` inside
warmed GPU containers on `.nwb.zarr` inputs (`benchmarks/sorters/results/`). MountainSort 5,
SpyKING CIRCUS 2, and Tridesclous 2 reflect the earlier 2026-11-09 `dspk-cli-sorter` run.
:::

Our sorters on two test recordings, compared head-to-head against upstream Python stacks
(Kilosort4 `v4.1.3`, SpikeInterface `v0.106.0`, EMUsort `v1.1.0`) on the same GPU.

![Head-to-head sorter scorecard comparing speed, memory, counts, and accuracy](/dsp-kitchen/benchmarks/sorter_comparison_table.png)

## Summary on the Neuropixels recording (`ZFM-02370`, 45 s, 383 ch)

| Sorter | Time (s) | Spikes | Units (good) | Good units: found / precision / accuracy (median) | Good units with accuracy ≥ 0.8 |
|---|---|---|---|---|---|
| `dspk kilosort4` (CUDA) | **15.8** (2.86× RT) | 128 325 | 283 (149) | 1.00 / 0.93 / 0.86 | 0.55 |
| `dspk kilosort4` (wgpu) | **23.6** (1.91× RT) | 128 416 | 278 (145) | 1.00 / 0.95 / 0.86 | 0.60 |
| Upstream `Kilosort4 4.1.3` (CUDA) | 112.4 (0.40× RT) | 137 374 | 263 (126) | reference (0.90 F1 vs saved Phy) | — |
| `SpikeInterface 0.106.0` KS4 (CUDA) | 114.8 (0.39× RT) | 137 374 | 263 (126) | 1.00 / 1.00 / 1.00 (vs live KS4) | 1.00 |
| MountainSort 5 (wgpu, 2026-11-09)¹ | 646 | 34 942 | 167 (134) | 0.84 / 0.89 / 0.61 | 0.37 |
| SpyKING CIRCUS 2 (wgpu, 2026-11-09)¹ | 47 | 120 469 | 452 (249) | 0.99 / 0.93 / 0.70 | 0.43 |
| Tridesclous 2 (wgpu, 2026-11-09)¹ | 45 | 83 110 | 562 (371) | 0.98 / 0.98 / 0.80 | 0.50 |

Kilosort4 is the reference here, so these measure agreement with it, not accuracy against ground
truth. SpyKING CIRCUS 2 and Tridesclous 2 split units more than Kilosort4 (452 and 562 units).

¹ Kilosort4 rows match spikes within ±0.4 ms and 40 µm against live Kilosort4 `4.1.3`
([Kilosort4 validation](/dsp-kitchen/plugins/synapse/validation/kilosort4/)); the 2026-11-09 rows match
within 60 µm against `saved_results/`.

## Setup

| | |
|---|---|
| Device | NVIDIA GeForce RTX 2070 with Max-Q Design (8 GB VRAM) |
| Host | Intel Core i7-9750H (12 threads) · Linux `7.3.4-300.fc45` |
| Runtime | Pre-autotuned CUDA and wgpu (Vulkan), `f32`, `reproducible = true` |
| Upstreams | Direct `kilosort 4.1.3`, `spikeinterface 0.106.0`, `emusort 1.1.0` (`torch 2.5.1+cu124`) |

Times are the whole program run from launch to output flush (warmed repeat 2). **Warm** means
all CubeCL, NVRTC/Vulkan, Numba, and PyTorch caches were populated before timed runs.

| Recording | Channels | Duration | Rate | Data |
|---|---|---|---|---|
| Neuropixels 1.0, `ZFM-02370_mini.nwb.zarr` | 383 | 45 s | 30 kHz | `.datasets/nwb/` |
| HD-EMG 4 × 8 grid, `15-25-33_meps.nwb.zarr` (`HDEMG`), first 200 s | 32 | 200 s | 24.4 kHz | `.datasets/nwb/` |

Details: [Kilosort4](/dsp-kitchen/plugins/synapse/validation/kilosort4/), [EMUsort](/dsp-kitchen/plugins/synapse/validation/emusort/), [the other sorters](/dsp-kitchen/plugins/synapse/validation/other-sorters/).
