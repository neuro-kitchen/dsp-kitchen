# Why matrix multiplication matters

## The problem with the obvious kernel

Take the channel covariance of a window: `C = X Xᵀ / n` for `X` with 384 channels and 60 000
samples. The obvious GPU kernel gives each output `C[i, j]` its own thread, which reads rows `i`
and `j` once and sums their products. It is correct, parallel and **slow**: 706 ms on an RTX 2070.

The reason is **arithmetic intensity**: how many operations a kernel does per byte it reads.
Each thread reads 2 × 60 000 floats to do 60 000 multiply-adds, one operation per 8 bytes. In
total the kernel reads every sample of `X` once per output it contributes to, 384 times, so
memory traffic, not arithmetic, sets the time. A GPU does far more arithmetic per second than it
can feed from memory (tens of operations per byte loaded), so a kernel at one operation per
8 bytes uses a few percent of the chip.

A **tiled** matrix product fixes this. A block of threads loads a tile of `X` (say 64 rows × 32
samples) into fast on-chip shared memory once, and every thread of the block reuses it for many
outputs; each thread also keeps a small block of outputs in registers. A value read from memory
is now used tens of times. The same covariance takes **18.9 ms** (37× faster) through a tuned
matrix product on the same GPU.

Writing a product that reaches this speed on every GPU and on the CPU is a project of its own:
tile sizes depend on register count, shared memory and the warp width; loads must be vectorized
and aligned; the CPU wants register-blocked loops instead of shared memory. So we do not write
it: we use [cubek](https://github.com/tracel-ai/cubek), Tracel's kernel library on CubeCL.
(CubeCL itself has no matrix product since 0.11; the rule we follow is to use a CubeCL function
where one exists, and cubek only for what CubeCL lacks.)

## Which DSP operations are matrix products

More than it first seems. A product `A · B` appears whenever **many outputs reuse the same
inputs with different weights**:

| Operation | As a product | Shape at sorter sizes |
|---|---|---|
| Channel covariance / whitening statistics | `X Xᵀ` | `[C, n] · [n, C]` |
| Spatial filters (whitening, Laplacian, referencing) | `W X` | `[C, C] · [C, n]` |
| Projections (PCA scores, template features) | `Wᵀ (X − μ)` | `[k, C] · [C, n]` |
| Filter banks / template matching | `T · im2col(X)` | `[templates, taps] · [taps, n]` |
| Gram matrices for k-means, nearest neighbours | `‖a‖² + ‖b‖² − 2 A Bᵀ` | `[n, d] · [d, m]` |

**And filtering?** A single FIR filter on one channel is a convolution: each output uses a few
inputs once, so it is memory-bound and a direct (tiled) convolution is the right kernel
(`dsp_base::filter::fir`). It becomes a product when **many filters share the same input**: a
bank of `k` filters over a channel is `[k, taps] · [taps, n]` (each input window, the column of
`im2col(X)`, is reused by all `k` filters). Kilosort4's universal-template detection is exactly
that: every channel correlated with every template. Spatial filtering (mixing channels at each
sample) is a product directly. So the question to ask of a filter is not "is it linear?" but
"how many outputs reuse each input?" The higher the reuse, the more a product wins.

## How `dsp_base::linalg::matmul` is built

`matmul::<F>(client, lhs, rhs, out, out_len)` computes `out[b] = lhs[b] · rhs[b]`.

**Views, not copies.** Operands are `MatrixView`s: a shape `[batches, rows, cols]`, strides and a
start offset over a device buffer. A transpose swaps two strides; a column range moves the
offset. Before the unsafe binding, every view checks that the highest element it reaches is in
the buffer and that the buffer fits 32-bit indexing.

**Exact routines only.** cubek offers tensor-core routines that round `f32` inputs to `tf32`
(about three significant digits). That is fine for neural networks and wrong for a covariance that
feeds a whitening matrix. The candidates are the routines that compute in `F`: `SimpleUnit`,
`DoubleUnit` (GPUs), `Gemm` and `CpuGemm` (written for the CPU runtime). After each launch the
wrapper reads the element types cubek actually used and rejects the routine if any differs from
`F`, so a future routine that promotes cannot slip in.

**Tuned per device.** cubek's own `Strategy::Auto` does not tune: it tries tensor cores, then
falls back to `SimpleUnit`, so on the CPU runtime it never reaches the CPU kernels. The wrapper
registers the exact candidates with CubeCL's autotuner instead, keyed by element type, size class
(powers of two) and memory layout; candidates a device cannot run report `Unavailable` and drop
out. The first call of each class compiles and times every candidate (seconds); later calls reuse
the choice.

**Alignment.** A device binding must start at a multiple of the device's offset alignment
(`client.properties().memory.alignment`; 32 bytes on the RTX 2070, up to 256 elsewhere). A
view whose offset is not aligned is copied once into a contiguous buffer (`MatrixView::aligned`,
a one-pass gather kernel that applies the offset itself).

## What it bought, and where it loses

Release build, median of 5, ms (`bench_linalg`). "cubek" = tuned exact cubek routines only;
"+ direct" = the same tuner with our direct kernel as one more candidate.

| Case | RTX 2070: before → cubek → + direct | Intel UHD 630: before → + direct |
|---|---|---|
| covariance 384 ch × 60 000 | 706 → 18.9 → **10.3** | 1795 → 1609 |
| dense spatial 384 × 384 · 384 × 60 000 | 86.6 → 8.65 → **8.82** | 1485 → 1453 |
| covariance 64 ch | 7.38 → 2.23 → 3.25 | 60.2 → **15.4** |
| dense spatial 64 ch | 2.70 → 2.40 → **1.09** | 45.1 → 44.3 |
| covariance 4 ch | 0.53 → 1.09 → 0.90 | 1.44 → 2.08 |
| dense spatial 4 ch | 0.078 → 2.25 → 0.169 | 0.82 → 1.35 |
| projection 61 → 6, 100 000 clips | 0.64 → 7.86 → 0.97 | 6.22 → 8.07 |

What the numbers teach:

- **Large products** gain 10–70× on the discrete GPU: this is where reuse pays.
- **Skinny products lose with a tiled library alone** (a 4-row or 6-row output has little to reuse,
  and each call carries 1–2 ms of setup). Adding the direct kernel as a tuning candidate gives most
  of it back: the tuner picks it for those shapes. **Measure; do not assume the library wins.**
- **What remains in small cases is the passes around the product**, not the product: centring into
  a copy, the split-major copy, summing the splits. At 4 channels these launches cost more than the
  product itself. A fused kernel (centre while multiplying) would remove them; the copies are kept
  because they make the product exact and its layout safe.
- **The integrated GPU gains little at 384 channels**: the exact unit routines do not suit it. Its
  fast path would be the tensor-style routines we exclude for precision.
