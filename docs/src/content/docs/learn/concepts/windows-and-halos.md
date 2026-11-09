---
title: Windows and halos
description: "How long recordings are processed in pieces with the same result as the whole."
sidebar:
  order: 3
---

A long recording doesn't fit on a device at once, so it is processed in *windows*. The point of the
design is that the pieces give the same answer as the whole.

## Two ways to get there

**Stream.** Windows in order, with filter state carried from one to the next. The output equals
processing the whole recording. It needs causal filters, because a zero-phase filter looks into the
future.

**Windows with halos.** Each window is read with extra samples on each side, the *halo*, and those
extra samples are cut off afterwards. It works with zero-phase filters, and the halo is as long as
the filter needs to [settle](/dsp-kitchen/reference/python/ops-filter/#settling_samples).

```text
            halo        the window's own samples        halo
        ┌─────────┬──────────────────────────────┬─────────┐
 read:  │/////////│                              │/////////│
        └─────────┴──────────────────────────────┴─────────┘
 kept:            └──────────────────────────────┘
```

| Mode | Used when | Result |
|---|---|---|
| `stream` | Every stage is causal | Equals the whole recording |
| `windows` | Some stage needs future samples | Within the settling tolerance of the whole recording |
| `auto` (default) | Picks `stream` where the stages allow it, else `windows` | |

The window size affects memory and speed, not the answer. The default is 65,536 samples per channel.

## Three threads at once

A run overlaps reading, the device and writing. The next window is read while this one is on the
device, and the writer has a queue of two windows, so memory stays bounded.

To find what limits a run, read `timings` in its report (`dspk run --json`, or the Python report).
The largest of `read_wait`, `device` and `encode + disk` is the bottleneck.

## Your own functions

[Custom operations](/dsp-kitchen/learn/python/custom-operations/) use the same idea: you declare how
far your function looks, and the windows are read with that much context.
