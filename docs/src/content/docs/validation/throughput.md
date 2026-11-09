---
title: Speed on real recordings
description: "Multiples of real time for whole files, read and processed."
sidebar:
  order: 3
---

From the benchmark of release 0.0.4 (2026-10-29): read and process, nothing written, in multiples of real
time, on an RTX 2070 Max-Q and an i7-9750H.

| Recording | Chain | CUDA | wgpu | CPU runtime |
|---|---|---|---|---|
| Neuropixels, 385 channels | zero-phase + CMR | 12.5× | 10.1× | 0.4× |
| Neuropixels, 385 channels | causal + CMR | 10.5× | 8.5× | 0.7× |
| HD-EMG, 32 channels, 300 s | zero-phase + CMR | 56× | 45× | 13× |
| HD-EMG, 32 channels, 300 s | causal + CMR | 76× | 59× | 38× |

With a Zarr file written (`dspk run`), Neuropixels dropped to 2–4×.

:::note[These numbers are from before two improvements]
Since then, vector kernels handle [odd row lengths](/dsp-kitchen/gpu/vector-widths/), which is what held
the CPU runtime back on 385 channels, and the writer runs on its own thread and compresses in parallel.
The benchmark hasn't been re-run yet, so expect these numbers to be pessimistic.
:::

## Latency

Streaming on CUDA, 1 ms of Neuropixels took 0.11 ms, and 10 ms took 0.36 ms. wgpu has a floor of about
0.4 ms per chunk, from submission overhead. CPU p99 jitter was up to about 10 times the median.

## Device time alone

Data already on the device, autotuned, one block of 384 channels × 60,000 samples (2 s at 30 kHz):

| Operation | CUDA | wgpu (Vulkan) | CPU runtime |
|---|---|---|---|
| IIR band-pass, 5th order, forward | 5.3 ms | 12.3 ms | 39 ms |
| FIR, 63 taps, causal | 3.1 ms | 6.4 ms | 33 ms |
| Running median, width 9 | 0.59 ms | 0.69 ms | 20 ms |
| Common median reference | 10.3 ms | 16.5 ms | 239 ms |
| Common average reference | 1.0 ms | 1.2 ms | 11 ms |
| `resample_poly` 1 / 3 | 1.4 ms | 1.7 ms | 16 ms |

Reproduce with `cargo test -p dspk-benchmarks --profile validate --test bench_kernels -- --ignored --nocapture`.
