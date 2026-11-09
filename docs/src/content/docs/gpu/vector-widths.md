---
title: Vector widths and odd row lengths
description: "Why a row of 385 fell back to scalar code, and how to use vectors anyway."
sidebar:
  order: 10
---

On the CPU, kernels get fast by processing 2 to 16 values at once: the CPU runtime turns CubeCL's
`Vector<F, N>` into SIMD. The FIR, resampling, median and CAR/CMR kernels gained 4 to 10 times this way.

## The catch

An aligned vector kernel reads and writes `&[Vector<F, N>]`. Row `r` starts at vector `r · len / N`, which
only exists when `N` divides the row length.

Real data doesn't cooperate. Neuropixels has 385 channels. A zero-phase window of 65,536 samples plus
two settling lengths of 349 is 66,234 samples, which only 2 divides.

So on real recordings the vector kernels were skipped, and the CPU fell back to scalar code. The 0.0.4
release benchmark showed it: 0.4× real time on Neuropixels with a zero-phase filter.

## Two fixes

**Store lane by lane at the end of a row.** The unaligned kernel keeps the same arithmetic on
`Vector<F, N>`, but gathers its inputs and scatters its outputs lane by lane. The last vector of a row
runs partly past the end, and only its valid lanes are stored. Inputs past the end repeat the last valid
lane, so every read stays in bounds. See `crates/dspk-ops/src/core/kernels/lanes.rs`.

**Pad for lockstep.** The IIR kernel advances `N` channels in lockstep, along time. There the channel
count is padded up to a multiple of `N` with zero channels, and the extra lanes are never read back.

Lanes are independent, so the stored lanes are **bit-identical** to the scalar kernels.

## The lesson

Benchmarks on convenient sizes can flatter you. Use the shapes real data has.
