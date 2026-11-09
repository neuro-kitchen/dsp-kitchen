---
title: Rust API
description: "A map of the dsp-kitchen crates, linking to their documentation on docs.rs."
sidebar:
  label: Overview
  order: 0
---

The full Rust API is on docs.rs. This page says where to start. To get everything through one
dependency, use [`dspk`](https://docs.rs/dspk), which re-exports the crates below as modules.

| Crate | What it is |
|---|---|
| [`dspk`](https://docs.rs/dspk) | Entry crate: re-exports the others; features choose what's included |
| [`dspk-core`](https://docs.rs/dspk-core) | Recordings, buffers, exact time and sample rates, windows, compute targets |
| [`dspk-io`](https://docs.rs/dspk-io) | Raw binary and Zarr recordings, format detection, chunk cache |
| [`dspk-ops`](https://docs.rs/dspk-ops) | Filters, resampling, spatial operators, linear algebra, statistics, peaks, device pipelines |
| [`dspk-pipeline`](https://docs.rs/dspk-pipeline) | Pipelines as data: plan, many inputs, merges, provenance |
| [`dspk-view`](https://docs.rs/dspk-view) | Min/max envelopes and pyramids for viewing |
| [`dspk-stream`](https://docs.rs/dspk-stream) | Network sessions for continuous signals (QUIC + TLS, protobuf) |
| [`dspk-hub`](https://docs.rs/dspk-hub) | Verified downloads of published artifacts |
| [`dspk-synapse`](https://docs.rs/dspk-synapse) | Electrophysiology: formats, probes, detection, clustering, matching, metrics |
| [`dspk-synapse-ml`](https://docs.rs/dspk-synapse-ml) | Pretrained models and their runtime |
| [`dspk-sorter-kilosort`](https://docs.rs/dspk-sorter-kilosort) | Kilosort4 and EMUsort (GPL-3.0) |
| [`dspk-sorter-ms5`](https://docs.rs/dspk-sorter-ms5) | MountainSort 5 (Apache-2.0) |
| [`dspk-sorter-si`](https://docs.rs/dspk-sorter-si) | SpyKING CIRCUS 2 and Tridesclous 2 |
| [`dspk-app-core`](https://docs.rs/dspk-app-core) | The app's core and its plugin interface |

## In `dspk-ops`

| Need | Module | Start with |
|---|---|---|
| Chain preprocessing steps | `pipeline` | `Pipeline`, `PipelineStage` |
| Filters | `filter` | `FilterSpec`, `execute_filter` |
| Mix channels | `spatial` | `SpatialWhitening`, `SurfaceLaplacian` |
| Noise, percentiles, element-wise math | `math` | `estimate_noise_std`, `execute_channel_mad` |
| Matrix products, PCA, ICA | `linalg` | `PcaModel`, `matmul` |
| Peaks | `peaks` | `find_peak_candidates` |
| Resampling | `resampler` | `execute_resample_poly` |

## Conventions

- Data is a `[channels, samples]` buffer, channel-major.
- Functions take a CubeCL `Client`: `ComputeTarget::from_env()?.client()?` picks the runtime named by
  `DSPK_RUNTIME`, or else the first compiled in.
- Functions on device buffers start with `execute_`. Data comes first with its shape, then
  `sample_rate`, then one settings struct.
