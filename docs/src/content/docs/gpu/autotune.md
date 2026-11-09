---
title: "Autotune, and why we benchmark"
description: "What timing kernels on your device buys, and what it cost a live preview."
sidebar:
  order: 9
---

The fastest way to run a kernel depends on the device and the problem size. A GPU with lots of parallel
units wants many short walks; the CPU wants a few long ones. No constant is right everywhere.

So where a kernel has several exact implementations, CubeCL's *autotuner* times each one on your device
the first time it sees a problem class, keeps the fastest, and
[caches the choice](/dsp-kitchen/learn/concepts/kernel-cache/). Classes are keyed by size, rounded up to
a power of two, so a similar size reuses the result.

An IIR pass is a good example. It tries splitting time into 1, 4, 16, 64 or 256 blocks. One block is a
single sequential walk per channel; more blocks give more parallel units at the cost of a second pass
over the data.

## What it cost

Autotuning benchmarks on the data you hand it. In a batch run that's invisible. In the app's live
preview, it wasn't.

Zooming out made a new window size, so a new class, so the tuner timed every candidate on the full
window, including the single sequential block. Each new size stalled the preview for 75 to 100 seconds.

Two fixes, each at a different level:

- **Bound the benchmark.** A pass's cost per step doesn't change past a certain length. So passes longer
  than 2²⁰ steps are timed on their first 2²⁰ steps and share one result.
- **Never tune during a preview.** A preview pins its choices with `dspk_core::compute::pin_tuned_choices`
  (a thread-local guard), so it uses fixed launch choices.

## A pinned choice has to be good

Pinning made previews responsive, and then they took 2.6 s. The pinned default was 16 blocks, which is
right for 384 channels (6,144 parallel walks) and very wrong for 2 channels (32 walks on a GPU with
thousands of units).

The pinned choice now depends on the channel count: enough blocks to give about 2,048 parallel walks.
It depends only on the channel count and the signal length, not on the device's timings, so a pinned run
still reproduces. 2 channels get 256 blocks, and 384 get 16.

With that, and with nested previews no longer requesting extra margin at every level, a preview went
from 5.5 s to 1.5 s.

## The lesson

Autotuning is a measurement, and a measurement takes time. Decide where you can afford it: on first use
in a batch, yes; while someone is dragging a slider, no. And when you pin, the pin needs the same
thought the tuner would have given it.
