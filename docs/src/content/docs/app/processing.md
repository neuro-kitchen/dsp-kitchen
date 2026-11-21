---
title: How a view is processed
description: "What a view's chain computes, where, and how far its result is from processing the whole recording."
sidebar:
  order: 2
---

A view's chain (the Inspector's **Processing**) runs on the GPU over the window you look at. Its
result is what the same chain gives on the whole recording, within a stated bound. Nothing is
written until you ask for a file. [Numerical accuracy](/dsp-kitchen/validation/accuracy/) has the
bounds and the tests behind them.

## Where it runs

| Where you are | Where the chain runs |
|---|---|
| Desktop app | On this computer's GPU |
| A page connected to a server | On the server's GPU ("Processing on the server" in the status bar) |
| A page on its own | In the browser, on WebGPU |

The same code runs in all three (`dspk_pipeline::view`).

## A window and its context

A filter needs samples before and after the ones it outputs: its impulse response has to settle.
Each window is read with that context on both sides (the chain's *settling*), processed, and only
its interior is kept.

- **The bound.** A window's result differs from the whole recording's by at most `tol` of the
  input's peak, where `tol` is how far each filter's impulse response is allowed to have decayed.
- **What `tol` is.**
  - Views use `1e-3`.
  - Batch runs and sorts use one count of the stored data: `2⁻¹⁵` for 16-bit integers, float32's
    resolution for float files.
- **What doesn't need it.** Steps without memory (scale, rectify, per-sample maths) need no
  context. Short steps (Teager-Kaiser, a median, an FIR filter) need exactly their own length.

## Zoomed out

A zoomed-out view draws `[min, max]` per pixel column. The columns are reduced on the GPU and only
they come back: 32 channels × 1200 columns, never the samples behind them.

| Window | What draws it |
|---|---|
| Fits one block (about 20 s of 32 channels) | One block, processed and kept on the GPU. Pans within it reuse it. |
| Up to 4 blocks | Tiles, each processed with its own context and reduced to its columns. The next tile starts while the last one's columns are read. |
| Wider | The chain's **processed overview**, below |

## The processed overview

When a chain stops changing, meaning no other chain is asked for within 1.5 s, the whole recording
is processed once in the background into `[min, max]` per 256 samples.

- **What it does for you.** Every zoomed-out view of that chain is then drawn from it, at any width,
  with nothing processed.
- **The order it builds in.** It starts with the part on screen and works outwards, and views fill
  in as it's built.
- **Before it starts,** a very wide view shows the raw signal and says so in the Inspector.
- **Where it lives.** It's kept in RAM (see **Memory** below).
- **Its accuracy.** Each column reaches at most one bucket (256 samples) past its own edges.

## Editing a chain

- **Changing or adding an op** reuses what is already on the GPU.
  - The window stays uploaded. A chain with another first op over the same window reads nothing
    again.
  - While a parameter is dragged, the chain up to that op is kept, and each new value reruns only
    from there.
  - A chain that starts with another view's chain reads that chain's output on the GPU and runs
    only its own ops.
- **An invalid op,** such as a median of even width, is refused with the reason, before anything
  runs.

## Memory

**GPU.**
- The views keep at most half of the GPU's memory. The size comes from the runtime, from
  `nvidia-smi` (NVIDIA) or from the kernel's `mem_info_vram_total` (AMD). `DSPK_DEVICE_MEMORY`
  (`6GiB`, `512MiB`) sets it by hand.
- Before a block is allocated, room is made for it. Views off screen give theirs back first
  (their working buffers, then their results), least recently used first.
- A view that gave its memory back keeps its chain and is processed again when it's shown.

**RAM.**
- One limit for everything the app keeps to be fast, across every open recording:
  - decoded recording chunks;
  - processed overviews;
  - spike windows.
- **Settings → Memory** sets it.
  - Automatic is a quarter of the computer's RAM. In a browser, it's half of what the browser
    reports, at most 4 GiB.
  - On a server, `DSPK_HOST_MEMORY` sets it.
- Past the limit, the least recently used item goes: it's read or computed again when it's needed.

**If the GPU is lost** (out of memory, a driver reset), the server logs it once and processed views
run on the CPU, the same chains, until it restarts.

## Settings that change processing

| Setting | Default | What it does |
|---|---|---|
| Settings → Memory, `[compute] memory` | Automatic | RAM the app keeps for decoded chunks, overviews and spike windows |
| `DSPK_DEVICE_MEMORY` | Half the GPU | GPU memory the views may keep |
| `DSPK_HOST_MEMORY` | A quarter of the RAM | RAM limit where there are no settings (a server) |
| `DSPK_PREVIEW_TIMING=1` | Off | Each view's time per step, in the status line and timings |
