# Case study: Kilosort4 detection, measured

Universal-template detection runs on every window of a recording, so its cost per window is the
cost of a sort. This chapter follows one optimisation pass from the first measurement to the
last, because the measurements changed the plan twice.

Benchmark: `crates/dsp-synapse-ml/tests/bench_detection.rs`, a Neuropixels-sized window (384
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
copies the slice into a new vector, then into the transfer. Measured alone (`bench_upload.rs`) it
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
[Why matrix multiplication matters](matmul.md): 6 outputs reuse each input window. The kernel ran
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
