---
title: Custom operations
description: "Run your own function over a recording, window by window."
sidebar:
  order: 4
---

`dspk.custom` runs your own Python function over a recording of any length. dsp-kitchen reads the
windows, the next one on another thread while your function runs. Your function only ever sees a
NumPy window `(channels, samples)`, float32.

## Measure: one row of results per window

```python
import numpy as np
import dspk

rec = dspk.io.open("session.zarr")
table = dspk.custom.measure(rec, lambda x: {"rms": np.sqrt((x ** 2).mean(axis=1))}, window_s=1.0)

table["start_s"], table["stop_s"], table["rms"]   # rms is (windows, channels)
```

## Transform: a processed copy

Tell it how many samples your function looks away, the `halo`. Windows are read with that much
context on each side and trimmed back, so the result equals your function on the whole array.

```python
@dspk.custom.transform(halo=12)
def smooth(x):
    k = np.ones(25, dtype=np.float32) / 25
    return np.stack([np.convolve(c, k, mode="same") for c in x])

y = smooth.apply(rec, start=0, stop=30_000)        # one array in memory
smooth.write(rec, "smoothed.zarr")                 # the whole recording, window by window
```

:::caution
A halo that is too small gives wrong samples at the window edges, with no error. Set it to at least
how far your function reaches.
:::

If your function needs to know where it is, take a second argument: `fn(x, w)` gets the
[`Window`](/dsp-kitchen/reference/python/custom/#window), with its sample range and time.

Your function runs on the calling thread, one window at a time, in order, so it can keep state.

Full details: [`dspk.custom`](/dsp-kitchen/reference/python/custom/).
