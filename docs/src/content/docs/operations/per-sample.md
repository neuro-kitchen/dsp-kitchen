---
title: Per-sample math
description: "Operations that look at one sample at a time."
sidebar:
  order: 4
---

These change each sample on its own, so they need no context around a window. As pipeline stages they
are `dspk.ops.stage.<name>(...)` in Python, and spec nodes with the same name.

| Operation | Spec | Result |
|---|---|---|
| Scale | `{"op": "scale", "alpha": 2.0, "beta": 0.0}` | `alpha * x + beta` |
| Subtract baseline | `{"op": "subtract_baseline", "baseline": 10.0}` | `x - baseline` |
| Clamp | `{"op": "clamp", "min": -500.0, "max": 500.0}` | `clip(x, min, max)` |
| Rectify | `{"op": "rectify"}` | `abs(x)` |
| Square | `{"op": "square"}` | `x**2` |
| Square root | `{"op": "sqrt"}` | `sqrt(x)`, negative values taken as 0 |
| Log₁₀ | `{"op": "log10"}` | `log10(abs(x))`, magnitudes below `1e-12` taken as it |

In the app they are under **Transform**.

## Two common chains

**An EMG envelope:** rectify, then low-pass.

```python
from dspk.ops import Pipeline, stage

envelope = Pipeline([stage.iir("bandpass", (20.0, 450.0)), stage.rectify(), stage.iir("lowpass", 6.0)], 2_000.0)
```

**A moving RMS:** square, low-pass, square root.

```python
rms = Pipeline([stage.square(), stage.iir("lowpass", 10.0), stage.sqrt()], 2_000.0)
```

The app has both as built-in recipes.
