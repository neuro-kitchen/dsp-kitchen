---
title: GPU lessons
description: "How the device code is built and why, using the changes we made as worked examples."
sidebar:
  order: 0
---

These chapters explain how dsp-kitchen's device code is built, and **why** it is built that way. They
are for anyone who wants to write fast, correct GPU code for signal processing, not only call it.

| Lesson | What it teaches |
|---|---|
| [Why matrix multiplication matters](/dsp-kitchen/gpu/matmul/) | Which DSP operations are matrix products, and why a tiled one is so much faster |
| [Covariance](/dsp-kitchen/gpu/covariance/) | Split matrix products, and a bug only an odd length showed |
| [HDBSCAN](/dsp-kitchen/gpu/hdbscan/) | A quadratic algorithm on 500,000 waveforms, and a kernel that hung |
| [Exact median](/dsp-kitchen/gpu/selection/) | Radix select on the bits of a float |
| [When the review was wrong](/dsp-kitchen/gpu/eigen/) | A performance claim is a hypothesis |
| [Kilosort4 detection, measured](/dsp-kitchen/gpu/detection/) | 477 to 123 ms per window, and the fix that didn't matter |
| [Kilosort4 clustering and matching](/dsp-kitchen/gpu/clustering/) | Graphs and trees as GPU work; 158 s to 5 s |
| [Moving data to the device](/dsp-kitchen/gpu/data-movement/) | One crossing per window, in stored form |
| [Autotune, and why we benchmark](/dsp-kitchen/gpu/autotune/) | What timing kernels on your device buys, and what it cost a live preview |
| [Vector widths and odd row lengths](/dsp-kitchen/gpu/vector-widths/) | Why a row of 385 fell back to scalar code |
| [Pitfalls](/dsp-kitchen/gpu/pitfalls/) | The traps that cost us a failing test or a frozen run |

Timings in these lessons are from when each change was made, mostly on an RTX 2070 Max-Q. They show
the size of an effect. Current numbers are in [Validation](/dsp-kitchen/validation/).

## How we work

Every change follows the same loop. It's slower than "write it and try it", and it's why the
measurements here can be trusted.

1. **Read before running.** Find out from the code what a change touches and what can go wrong. Run a
   test to check a conclusion, not to find one.
2. **Baseline first.** Measure the current code with the same benchmark that will measure the new code,
   on a discrete GPU, an integrated GPU and the CPU runtime. A rewrite can win on one and lose on another.
3. **A host reference in `f64`.** Check device results against a plain host computation in double
   precision, never against the previous kernel, which may be wrong too.
4. **Odd sizes and `f64`.** Use odd lengths, because a layout bug hid behind even ones. Where the runtime
   supports `f64`, check the element type is kept end to end: an `f32` step shows up as a 1e-7 error
   where 1e-12 is expected.
5. **No silent constants.** Every tuning number is a named, documented `const` that says what it trades.
6. **Short runs.** A test or benchmark gets at most five minutes. A run that needs more is a sign the
   problem size or the algorithm is wrong.

## Generic over the device and the element type

Algorithms take a CubeCL `Client`, the device the caller chose, and are generic over the float type
`F: DspFloat`. The same source runs on CUDA, ROCm, wgpu and the CPU. Launch sizes come from the device's
own properties, and settings with no device-independent best value go to CubeCL's autotuner.
