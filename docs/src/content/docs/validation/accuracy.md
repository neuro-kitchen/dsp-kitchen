---
title: Numerical accuracy
description: "How close dsp-kitchen's float32 results are to float64 references, why, and what checks it."
sidebar:
  order: 2
---

dsp-kitchen computes in float32 on the GPU. Each statement below has a test that fails when it
stops being true. [What the tests check](/dsp-kitchen/validation/tests/) lists where they are.

## Recursive filters

- **Design:** in float64, on the host, equal to SciPy's `butter`, `cheby1`, `bessel` and
  `iirnotch` to `1e-9` per coefficient.
- **Computation:** in float32, as cascaded second-order sections in state-variable form. The plain
  direct form loses about `1e-2` of the signal's peak at low cutoffs; the state-variable form stays
  at float32's rounding floor.
  - Measured on a 0.5 Hz high-pass at 30 kHz (a cutoff ratio of 1.7e-5): `1.9e-5`–`2.0e-5` of the
    peak against float64, zero-phase and from a steady state.
- **Never in half precision.** Rounding coefficients to float16 or bfloat16 moves poles next to
  z = 1 out of the unit circle: a 4–8 Hz order-4 Butterworth gives NaN (braindecode #1657). Every
  section of every design in the sweep below stays stable rounded to float32.

### The sweep

Butterworth low- and high-passes at orders 1, 2, 4, 8, 12 and 16 and cutoff ratios from 1e-4 to
0.45, at 24,414 Hz, filtered on the GPU in float32 and compared with float64 on the host.

- **What fails it:** over `2e-4` of the peak up to order 8, or a section unstable in float32.
- **Orders 12 and 16** are reported in the table without a limit.
- **Results** (wgpu, NVIDIA RTX 2070 Max-Q), the largest error of the peak over zero-phase and
  steady-state passes:

  | | Cutoff ratio 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 to 0.45 |
  |---|---|---|---|---|---|
  | Low-pass, orders 1–8 | ≤ 5.8e-6 | ≤ 5.9e-6 | ≤ 2.8e-6 | ≤ 7.2e-7 | ≤ 7.2e-7 |
  | Low-pass, order 12 | **5.0e-3** | 4.8e-6 | 2.8e-6 | 1.6e-6 | ≤ 9.6e-7 |
  | Low-pass, order 16 | **1.2e-1** | **1.2e-1** | **3.2e-4** | 1.3e-6 | ≤ 1.7e-6 |
  | High-pass, orders 1–16 | ≤ 1.1e-5 | ≤ 2.8e-6 | ≤ 2.5e-6 | ≤ 5.1e-7 | ≤ 2.0e-7 |

  Up to order 8 every design stays within 6e-6 of the peak (−104 dB or better). Low-passes of order 12 and
  16 at cutoff ratios of 1e-3 and below lose float32 precision. A chain ending in such a low-pass runs it
  at a lower rate in views ([below](#a-low-pass-at-a-lower-rate)), where its cutoff ratio is about 0.1.
- **Run it:**

  ```sh
  DSPK_TEST_RUNTIMES=wgpu cargo test -p dspk-ops --test it -- accuracy:: --nocapture
  ```

  It prints the error of every design in dB.

## Windows against the whole recording

A window processed with `L` samples of context on each side differs from the whole recording by
the part of each filter's impulse response past `L`:

```text
|y − y_whole| ≤ max|x| · Σ_{k > L} |h[k]|   ≤ tol · max|x| · Σ_k |h[k]|
```

`L` is chosen so that the tail is below `tol` of the whole response.

| | `tol` | Error at the window's edges |
|---|---|---|
| Views | `1e-3` | ≤ 0.1 % of the input's peak (about 60 dB) |
| Batch runs and sorts, int16 data | `2⁻¹⁵` | below one count of the stored data |
| Batch runs and sorts, float data | `2⁻²³` | float32's own resolution |

**Tested:**
- chunked filtering against the whole recording, in stream and window mode;
- processed views against the same window processed whole, as one block and in tiles.

## A low-pass at a lower rate

A chain ending in a low-pass runs it at a rate `M` times lower
([how a view is processed](/dsp-kitchen/app/processing/)).

- **The decimating filter:** a Kaiser FIR (β 10, 16·M + 1 taps) cut at the new Nyquist. It's flat
  to 0.3 × the new rate and about 100 dB down from 0.7 × it on. Only energy that folds into the
  low-pass's band (0 to `fc`) matters, and all of it comes from 0.875 × the new rate or above. A
  Teager-Kaiser output is broadband, so that stop-band is what keeps its folded power below the
  envelope it carries.
- **Where its error comes from:**
  - that filter's pass-band ripple (below 1e-4);
  - its stop-band leakage (about 100 dB);
  - for zoomed-in samples, linear interpolation of a signal sampled at at least 8× its cutoff.
- **Tested:** band-pass → Teager-Kaiser → 20 Hz low-pass at 24.4 kHz, at the lower rate against the
  full rate.
  - **Zoomed in:** samples within 2 % of the signal's scale.
  - **Zoomed out:** each column's `[min, max]` lies between the full-rate signal's extremes over the
    column shrunk by 3 low-rate samples on each side and grown by 3. A column's edges in the low
    rate's samples fall up to 2 of them early, and its last sample up to one more before its edge.
    For this chain that's 3 × 128 = 384 samples (16 ms), well under one pixel column at the zooms
    that use it.

## Zoomed-out views and the processed overview

- **Columns** are `[min, max]` merges. A merge is associative and commutative, and an empty column
  changes nothing. So tiles, pyramid levels and overview buckets reduced apart and merged give the
  columns of the whole. Tested on 10,000 random triples.
- **A column from the processed overview** holds every sample of the column. It reaches at most one
  256-sample bucket past its edges. Tested against the view processed directly.

## Properties checked on the GPU

| Property | What it means |
|---|---|
| Linearity | `f(a·x₁ + b·x₂) = a·f(x₁) + b·f(x₂)` for a recursive filter, within `1e-4` of the peak |
| Time invariance | A delayed input gives the same output, delayed, and nothing before it |
| Zero phase | A forward-backward filter keeps a symmetric burst symmetric |
| Parseval | The real FFT under the long-FIR path keeps energy, within `1e-5` |
| Overlap-save | The FFT FIR equals direct convolution within `2e-5` of the scale |
| Fusion | Fused per-sample ops equal the ops run one by one |
