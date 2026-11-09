---
title: Operations
description: "Run operations on arrays, one by one or as a chain on the device."
sidebar:
  order: 2
---

## One at a time

Each call uploads its input, runs, and returns NumPy.

```python
import dspk

y = dspk.ops.filter.iir(x, 30_000.0, "bandpass", (300.0, 6000.0))
y = dspk.ops.spatial.cmr(y)
```

## As a chain

Two separate calls move the data to the device and back twice. A
[`Pipeline`](/dsp-kitchen/reference/python/ops/#pipeline) does one upload, every stage, one download.

```python
from dspk.ops import Pipeline, stage

chain = Pipeline([stage.iir("bandpass", (300.0, 6000.0)), stage.cmr()], 30_000.0)
y = chain.process(x)
```

## Chunks of a long signal

`stream=True` treats chunks as consecutive pieces of one signal. Filter state carries over, so the
output equals processing the whole signal at once. It needs causal filters:

```python
chain = Pipeline([stage.iir("highpass", 300.0, mode="forward"), stage.cmr()], 30_000.0, stream=True)
for chunk in chunks:
    y = chain.process(chunk)
```

Without `stream`, each chunk stands alone, and
[`settling`](/dsp-kitchen/reference/python/ops/#pipeline) says how much context it needs to match the
whole signal.

## Picking a runtime

Every call takes `runtime="cuda"` and so on, or set one for everything with `dspk.runtime.set`.

See [Operations](/dsp-kitchen/operations/) for what's available.
