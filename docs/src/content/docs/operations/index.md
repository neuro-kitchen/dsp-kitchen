---
title: Operations
description: "Every operation once: what it does, and how to call it from each tool."
sidebar:
  order: 0
---

Each page covers a family of operations: what they do, the SciPy call they match where there is one,
and how to run them from Python, Rust, a pipeline spec or the app.

| Family | Operations |
|---|---|
| [Filters](/dsp-kitchen/operations/filters/) | IIR, notch, SOS, FIR, Gaussian, median, Teager–Kaiser, template subtraction |
| [Referencing](/dsp-kitchen/operations/referencing/) | CAR, CMR, Laplacian, whitening |
| [Rate](/dsp-kitchen/operations/rate/) | `resample_poly`, `decimate` |
| [Per-sample math](/dsp-kitchen/operations/per-sample/) | scale, baseline, clamp, rectify, square, sqrt, log10 |
| [Channels](/dsp-kitchen/operations/channels/) | Channel maps, reordering, selecting |
| [Statistics and models](/dsp-kitchen/operations/statistics/) | Noise levels, windows, peak candidates, PCA |
| [Pipelines](/dsp-kitchen/operations/pipelines/) | Running chains over whole recordings |

## Everything runs on the device

Data is `(channels, samples)`. Operations run on the [runtime](/dsp-kitchen/learn/concepts/runtimes/)
you pick, and return `float32` of the same shape, unless the operation changes it.

## Conventions

- **Sample rate.** Design functions take it last; functions applied to data take it after the data.
  Stages take none, because the pipeline knows the sample rate.
- **Defaults** are shown in every signature, and follow SciPy where an equivalent exists.
- **Rust** examples below are stages for a
  [`Pipeline`](https://docs.rs/dspk-ops/latest/dspk_ops/struct.Pipeline.html). Each family also has
  `execute_*` functions on device buffers.

Exact parameters: [Python](/dsp-kitchen/reference/python/) and
[spec](/dsp-kitchen/reference/schemas/pipeline/) references.
