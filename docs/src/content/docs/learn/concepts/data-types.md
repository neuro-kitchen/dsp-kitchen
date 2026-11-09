---
title: Data types
description: "Recordings move in their stored type; results come out as float32."
sidebar:
  order: 2
---

A recording moves in the type it was stored in. The conversion to `float32` happens once, on the
device.

```text
int16 on disk  →  int16 in RAM  →  int16 uploaded  →  float32 on the device  →  float32 out
```

Why it matters: an `int16` recording moves half the bytes a `float32` copy would, at every step
where bytes are the bottleneck.

## What you get back

Outputs are `float32` for every input, so every recording processed comes out in the same form. It
matches how `scipy.signal` returns floats for integer input (scipy's are `float64`).

## Exceptions

| Input | What happens |
|---|---|
| `float64` | Converted to `float32` on the host: it halves the upload, and wgpu has no `f64` |
| Python lists and other NumPy dtypes | Become `float32` on the host first |
| Fits that read host data (`Whitening.fit_*`, `PCA`, `PPCA`) | `float32` on the host first |
| Synapse's in-memory detectors | `float32` on the host first |

The [Python reference](/dsp-kitchen/reference/python/) says so on each function where it applies.

## Writing integers

`dspk.io.write_raw` writes an integer array as the stored integers (`value = stored * gain`), and
its `format` defaults to the array's dtype, so nothing is converted.
