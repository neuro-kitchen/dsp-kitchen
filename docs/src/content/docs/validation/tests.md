---
title: What the tests check
description: "The rules the test suite enforces, and where."
sidebar:
  order: 2
---

| Claim | How it's checked | Where |
|---|---|---|
| Every operation is right | Against a host reference (SciPy or NumPy results, or a direct computation), on every runtime compiled in | `crates/dspk-ops/tests/kernel_runtimes.rs` |
| Windows don't change the answer | Chunked runs against the whole recording, in stream and window mode | `crates/dspk-ops/tests/filter_chunks.rs`, `engine/dspk-pipeline/tests/run.rs` |
| Stored integers change nothing | int16 recordings uploaded as integers give the same result as float | `engine/dspk-pipeline/tests/run.rs` |
| Vector kernels are exact | Each vector width against the scalar version | `crates/dspk-ops` kernel tests |
| Detection doesn't depend on batch size | The same spikes and locations at any batch size | `synapse/dspk-synapse/tests/streaming_invariance.rs` |
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
