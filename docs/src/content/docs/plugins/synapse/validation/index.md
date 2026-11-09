---
title: Validation
description: "Each sorter against Kilosort4's saved results, on one Neuropixels recording."
sidebar:
  label: Overview
  order: 0
---

:::note
Measured in October 2026 on the setup in the [overview](/dsp-kitchen/plugins/synapse/validation/). The scripts in `benchmarks/sorters` still use the previous Python API and don't run on the current package, so these results haven't been re-run since.
:::

Our sorters on two test recordings, compared with Kilosort4's own results where those exist.

## Summary on the Kilosort4 recording

| Sorter | Time (s) | Spikes | Units | KS4 good: found / precision / accuracy (median) | KS4 good with accuracy ≥ 0.8 |
|---|---|---|---|---|---|
| Kilosort4 (ours)¹ | — | 130 551 | 291 | 1.00 / 0.95 / 0.82 | 0.52 |
| MountainSort 5 | 404 | 35 121 | 165 | 0.83 / 0.91 / 0.59 | 0.36 |
| SpyKING CIRCUS 2 | 44 | 120 469 | 452 | 0.99 / 0.93 / 0.70 | 0.43 |
| Tridesclous 2 | 41 | 83 110 | 562 | 0.98 / 0.98 / 0.80 | 0.50 |

Kilosort4 is the reference here, so these measure agreement with it, not accuracy against ground
truth. SpyKING CIRCUS 2 and Tridesclous 2 split units more than Kilosort4 (452 and 562 units).

¹ Matched within 40 µm (the other rows: 60 µm), from the [Kilosort4 page](/dsp-kitchen/plugins/synapse/validation/kilosort4/).

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
| Neuropixels 1.0, `ZFM-02370_mini.imec0.ap.short.bin` | 383 | 45 s | 30 kHz | `.datasets/kilosort4/` |
| HD-EMG 4 × 8 grid, `15-25-33_meps.nwb.zarr` (HDEMG series), first 200 s | 32 | 200 s | 24.4 kHz | `.datasets/nwb/` |

Details: [Kilosort4](/dsp-kitchen/plugins/synapse/validation/kilosort4/), [EMUsort](/dsp-kitchen/plugins/synapse/validation/emusort/), [the other sorters](/dsp-kitchen/plugins/synapse/validation/other-sorters/).

Ground-truth accuracy, and a comparison with SpikeInterface on the same machine, aren't done yet.
