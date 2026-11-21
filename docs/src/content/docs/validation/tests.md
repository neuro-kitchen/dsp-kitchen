---
title: What the tests check
description: "The rules the test suite enforces, and where."
sidebar:
  order: 2
---

| Claim | How it's checked | Where |
|---|---|---|
| Every operation is right | Against a host reference (SciPy or NumPy results, or a direct computation), on every runtime compiled in | `crates/dspk-ops/tests/it/kernel_runtimes.rs` |
| Filter designs equal SciPy | Coefficients to `1e-9`, filtering to SciPy's `sosfilt` and `sosfiltfilt` | `crates/dspk-ops/tests/it/filter_scipy.rs` |
| float32 holds across designs | Orders 1–16, cutoff ratios 1e-4–0.45, against float64; every section stable rounded to float32 | `crates/dspk-ops/tests/it/accuracy.rs` |
| The maths' own properties | Linearity, time invariance, zero phase, Parseval | `crates/dspk-ops/tests/it/properties.rs` |
| Windows don't change the answer | Chunked runs against the whole recording, in stream and window mode | `crates/dspk-ops/tests/it/filter_chunks.rs`, `engine/dspk-pipeline/tests/run.rs` |
| Views equal the whole | A processed view, as one block or in tiles, against the window processed whole; an edited chain against one made fresh; a branch against its whole chain | `engine/dspk-pipeline/src/view/processed.rs` |
| A low-pass at a lower rate is the same low-pass | The chain at the lower rate against the full rate, zoomed out and in | `engine/dspk-pipeline/src/view/processed.rs` |
| Overviews hold what views show | The processed overview against the view processed directly; `[min, max]` merges associative and commutative | `engine/dspk-pipeline/src/view/processed.rs`, `view/dspk-view/src/envelope/fold.rs` |
| Memory stays within its limits | GPU memory kept within the device budget over wide windows and many chains; the RAM cache within its limit over any number of recordings | `engine/dspk-pipeline/src/view/budget.rs`, `crates/dspk-core/src/memory.rs` |
| Stored integers change nothing | int16 recordings uploaded as integers give the same result as float | `engine/dspk-pipeline/tests/run.rs` |
| Vector kernels are exact | Each vector width against the scalar version | `crates/dspk-ops` kernel tests |
| Detection doesn't depend on batch size | The same spikes and locations at any batch size | `synapse/dspk-synapse/tests/it/streaming_invariance.rs` |
| Python matches SciPy | The Python test suite, with SciPy as reference | `bindings/python/tests` |
| The documented examples work | Every docstring example runs as a doctest | `bindings/python/python/dspk` |
| Stubs match the extension | A test fails when they disagree | `bindings/python/tests/test_stubs.py` |

## Rules for writing tests

- Compare against a host computation in `f64`, never against the previous kernel.
- Use odd sizes: a layout bug once hid behind even ones.
- Where the runtime supports `f64`, check the element type is kept end to end.

## What CI covers

GitHub runners have no GPU, so CI runs the CPU runtime only. CUDA and wgpu are compiled in but not
exercised there, and the full suite on them runs locally on a GPU machine before a release.

CI also checks dependencies and licences (`cargo-deny`), that the generated Python reference is
current, and that this site builds.
