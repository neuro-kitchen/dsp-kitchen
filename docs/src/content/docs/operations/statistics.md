---
title: Statistics and models
description: "Noise levels, windows, peak candidates and PCA."
sidebar:
  order: 6
---

These answer a question about the signal, rather than producing another signal, so they're called
from Python or Rust and aren't pipeline stages.

## Per-channel statistics

| Function | What it gives |
|---|---|
| [`channel_noise_std`](/dsp-kitchen/reference/python/ops-math/#channel_noise_std) | `median(abs(x)) / 0.6745`: robust noise, barely moved by the spikes |
| [`channel_mad`](/dsp-kitchen/reference/python/ops-math/#channel_mad) | `median(abs(x - median(x)))`, unscaled |
| [`channel_mean_std`](/dsp-kitchen/reference/python/ops-math/#channel_mean_std) | Mean and population standard deviation |

## Windows

[`dspk.ops.math.window`](/dsp-kitchen/reference/python/ops-math/#window): Hann, Hamming, Gaussian and
Kaiser, symmetric.

## Peak candidates

[`dspk.ops.peaks.candidates`](/dsp-kitchen/reference/python/ops-peaks/#candidates) finds local extrema
reaching a height per channel. A flat peak is reported at its first sample.

```python
noise = dspk.ops.math.channel_noise_std(x)
peaks = dspk.ops.peaks.candidates(x, 4 * noise, polarity="negative")
```

## PCA and probabilistic PCA

[`PCA`](/dsp-kitchen/reference/python/ops-linalg/#pca) and
[`PPCA`](/dsp-kitchen/reference/python/ops-linalg/#ppca) fit on the device. Channels are the features,
and samples the observations.

```python
pca = dspk.ops.linalg.PCA.fit(x, 3)
scores = pca.transform(x)        # (3, samples)
```

## Crossing a threshold

There's no general threshold-crossing or onset operation in the core yet. For spikes, see
[detection](/dsp-kitchen/plugins/synapse/learn/detection/) in the Synapse plugin.
