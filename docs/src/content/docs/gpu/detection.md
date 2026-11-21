---
title: "Kilosort4 detection, measured"
description: "477 to 123 ms per window, and why the planned fix was not the one that mattered."
sidebar:
  order: 6
---

Universal-template detection runs on every window of a recording, so its cost per window is the
cost of a sort. This chapter follows one optimisation pass from the first measurement to the
last, because the measurements changed the plan twice.

Benchmark: `a detection benchmark on the Kilosort4 detector`, a Neuropixels-sized window (384
channels in two columns, 60 000 samples, 1149 template centres), whitened noise with injected
spikes, on an RTX 2070. Every change below keeps the detected spikes identical (4944).

## 1. Measure the whole, then the parts

The task on the list was "remove the per-window CPU↔GPU stalls": detection read the candidate
counts back before sizing its outputs, then made six separate downloads. The whole window took
**477 ms**. Timing each step (a device sync after each kernel, then removing the probe):

| step | ms |
|---|---|
| upload | 32 (device side) |
| correlate every channel with every template | 32 |
| `centre_response` | **165** |
| `neighbour_max` | **67** |
| local peak score | 12 |
| candidate count + read + write, features, downloads | 7 |

The stalls were 7 ms of 316. And the steps did not add up to 477: about 160 ms were spent
**outside** the kernels. Splitting the upload into "host call returns" and "transfer done" showed
114 ms on the host thread for a 92 MB window.

## 2. The upload was the largest cost

`buffer::upload` used `client.create_from_slice`, which CubeCL documents as the slower path: it
copies the slice into a new vector, then into the transfer. Measured alone (an upload benchmark) it
moved 0.5 GB/s, against about 1.2–1.6 GB/s for `client.create(Bytes)` with owned bytes, on wgpu and
CUDA alike. Two changes:

- Uploads hand CubeCL owned bytes, and the pipeline writes a **persistent** device buffer in place
  (`client.write`) instead of allocating one per window.
- The **read-ahead thread** that reads the next window from disk now allocates it as an owned
  buffer (`WindowLoader::stream_owned_while`), which moves into the upload: the main thread copies
  nothing. Its upload call now takes ~0 ms; the remaining 30 ms are the transfer itself.

## 3. `centre_response`: reuse in registers

Each (centre, sample) combines 10 channels × 6 templates for each of 5 template sizes: 300 reads of
the correlation buffer. But a channel-template value is the same for every size, so it was read 5
times. Per unit the work is a small matrix product, `R[s, k] = Σ_c W[s, c] · B[c, k]`. With the
sizes as compile-time constants the loops unroll, the 60 values of `B` and the 10 channel indices
stay in registers, and the reads drop 5×: 165 → ~33 ms. Sums and the maximum search keep their old
order, so ties resolve as before.

## 4. Correlation: a filter bank through shared memory

Correlating every channel with every template is the filter bank of
[Why matrix multiplication matters](/dsp-kitchen/gpu/matmul/): 6 outputs reuse each input window. The kernel ran
one thread per (channel, template) and read 61 samples each from memory. Now a block loads its
stretch of each channel (plus 60 samples of context) and the 6 × 61 templates into shared memory
once, and each thread computes all 6 outputs: 32 → ~1 ms.

## 5. `neighbour_max`: read a block's neighbourhood once

Each (centre, sample) takes the maximum over its 100 nearest centres: 27.6 GB of reads per window,
at the GPU's memory bandwidth. Faster arithmetic cannot help; only fewer reads can. Consecutive
centres lie next to each other on the probe and share most of their neighbours, so the host groups
them into blocks and lists each block's **union** of neighbour rows once. A cube loads that union
for its samples into shared memory (a few global reads per output instead of 100) and each thread
takes its maximum there. A maximum does not depend on order, so the result is exact: 67 → ~40 ms.

## Result

| | before | after |
|---|---|---|
| upload (main thread) | ~200 ms | 30 ms |
| detection kernels | ~316 ms | 93 ms |
| **per window** | **477 ms** | **~123 ms** |

The stall removal that started the pass was not done: with capacity-sized output buffers,
overflow re-runs and double-buffered results it would save ~7 ms of ~123. **Measure before you
build.**

## 6. Eliminating WGSL register spilling in `centre_response` and drift detection

Full-pipeline container benchmarks against upstream Kilosort4 `4.1.3` uncovered a stark backend
gap on 383-channel Neuropixels recordings: while CUDA finished in `17.5 s`, `wgpu` took
`250.5 s` — with **95.2%** of that time (`119.53 s` in `Computing drift` and `118.61 s` in
`Detecting spikes`) inside `UniversalDetector`.

Two causes dominated `wgpu` (Vulkan/WGSL):

1. **Per-thread local arrays in `centre_response_kernel`.** Each thread allocated
   `chan[n_chans]`, `bv[n_chans]`, and `resp[n_sizes * n_templates]` (`var<function>` arrays in
   WGSL). Across 383 channels and 1532 centres, NVIDIA's SPIR-V compiler spilled those indexed
   local arrays into per-thread scratch memory, slowing `centre_response` by >20× vs CUDA.
2. **Repeated PCA projections and kriging matrix builds.** `STAGE_DRIFT` computed full `wPCA`
   projections only to discard them during drift estimation, and `DriftCorrector` rebuilt and
   multiplied `K(y − shift, x) · (K(x, x) + 0.01·I)⁻¹` on every window of both `STAGE_DETECTION`
   and `STAGE_MATCHING`.

Five bit-exact changes eliminated the bottleneck (verified to `0` byte differences across all 56
output arrays on both CUDA and wgpu):

- **Shared-memory staging in `centre_response_kernel`:** `tile_y` centres collaboratively load
  their `ic` (`ic_s`) and `weights` (`w_s`) tables into workgroup shared memory once before
  `sync_cube()`. Eliminating `chan` and `resp` arrays and tracking only scalar
  `(best, best_arg, best_s)` in `(k, s)` order removes WGSL register spilling while preserving
  exact `(s, k)` tie-breaking (`mag > best || (mag == best && s < best_s)`).
- **Largest `tile_y` with guarded fusion:** `NeighbourTiles::new` maximizes unfused `tile_y`
  first and enables fused `neighbour_max_score_kernel` only when the larger fused workgroup fits
  shared memory, while `centre_response_launch` bounds `tile_y` to `max_shared_memory_size`.
- **Reusable device output buffers and prefix readback:** `picked_buf`, `feat_buf`, and `amp_buf`
  are retained with geometric growth and read back via `buffer::download_prefix` for the exact
  `n` candidate spikes.
- **Amplitude-only `spike_amplitudes_kernel` for drift:** `STAGE_DRIFT` calls `detect_for_drift`,
  skipping `x` / `wpca` feature dot products while matching `sample`, `centre`, `amplitude`,
  `template`, `size`, and position bit-for-bit.
- **Precomputed per-batch kriging matrices (`DriftCorrector`):** all batch `[channels, channels]`
  matrices `M_b` are computed once at construction (up to `256 MiB`, falling back on demand for
  larger recordings), reducing per-window drift correction to a single `M_b · X` product.

Measured impact on the 45 s 383-channel Neuropixels recording (`dspk 0.0.13`, warmed run 2):

| Stage / Total | `v0.0.13` pre-opt (`wgpu`) | After (`wgpu`) | Speedup (`wgpu`) | After (`cuda`) |
|---|---:|---:|---:|---:|
| `Computing drift` | `119.53 s` | **`5.44 s`** | **21.95×** | `4.69 s` (from `5.86 s`) |
| `Detecting spikes` | `118.61 s` | **`5.75 s`** | **20.64×** | `4.26 s` (from `5.41 s`) |
| **Whole run (e2e)** | `250.53 s` | **`23.56 s`** | **10.63×** | **`15.76 s`** (`1.11×`) |

Both backends now beat upstream PyTorch CUDA Kilosort4 `4.1.3` (`112.42 s`): **7.13× faster** on
CUDA and **4.77× faster** on `wgpu`.
