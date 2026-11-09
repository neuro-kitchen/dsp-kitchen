---
title: Runtimes
description: "The same kernels on CUDA, wgpu, HIP and the CPU."
sidebar:
  order: 1
---

Every operation in dsp-kitchen is a *kernel* written once, in [CubeCL](https://github.com/tracel-ai/cubecl).
A runtime is what runs it.

| Runtime | Runs on | Tested on |
|---|---|---|
| `wgpu` | Any GPU, through Vulkan, Metal or DirectX 12 | NVIDIA RTX 2070 Max-Q, Intel UHD 630 (Vulkan) |
| `cuda` | NVIDIA GPUs | NVIDIA RTX 2070 Max-Q |
| `cpu` | The CPU, through MLIR and LLVM | x86-64 |
| `hip` | AMD GPUs (ROCm) | Compiles; **untested**, for lack of an AMD GPU |

You pick the runtime when the program runs. Nothing is rebuilt.

## The results agree

The kernel is the same on every runtime, and each is tested against a host reference on every
runtime compiled in. So you can develop on a laptop's CPU and run the same chain on a server's GPU.

Where a kernel has several exact implementations, the runtime's autotuner times them on your device
and keeps the fastest ([the kernel cache](/dsp-kitchen/learn/concepts/kernel-cache/) remembers the
choice). The vector versions are bit-identical to the scalar ones; a few other choices (FIR tiles, IIR
time blocks) agree to rounding.

## One limit to know

Kernels index with 32-bit integers, because WebGPU has no 64-bit integers and one kernel source
serves every runtime. So one window or batch on the device holds at most 2³² − 1 values, for example
384 channels × 11.1 million samples. Larger requests are refused with an error, and
[windows](/dsp-kitchen/learn/concepts/windows-and-halos/) keep real runs far below it.
