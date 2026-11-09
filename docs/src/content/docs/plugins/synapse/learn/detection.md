---
title: Detection
description: "Find spikes, on arrays or over a whole recording."
sidebar:
  order: 2
---

## On a whole recording

```python
import dspk

rec = dspk.io.open("session.imec0.ap.bin")
probe = dspk.synapse.io.probe_of("session.imec0.ap.bin")
sorting, summary = dspk.synapse.detection.detect_recording(rec, probe)

summary["spikes"], summary["sigmas_uv"]
sorting.spikes()["locations"]      # where each spike came from, in µm
```

It reads in place in batches: optional CAR, zero-phase band-pass, noise measured on chunks spread over
the recording, threshold crossings, removing the same spike seen on neighbouring channels, snippets and
templates. Chunked detection gives the same result as detecting on the whole recording.

The command line does the same:

```sh
dspk detect session.imec0.ap.bin --output detected.sorting.zarr
```

## On data in memory

| Detector | Finds |
|---|---|
| [`threshold`](/dsp-kitchen/plugins/synapse/reference/python/detection/#threshold) | Local extrema beyond `threshold * sigma` on each channel |
| [`neo`](/dsp-kitchen/plugins/synapse/reference/python/detection/#neo) | Peaks of the Teager–Kaiser energy |
| [`matched_filter`](/dsp-kitchen/plugins/synapse/reference/python/detection/#matched_filter) | Correlation with a spike shape |

Noise is `median(|x|) / 0.6745` (Quiroga et al., 2004) unless you give `sigmas`. Each returns `samples`,
`channels` and `amplitudes`, sorted by time, and reports the spike's extremum.

`spacing="locally_exclusive"` (the default) keeps a spike unless a larger one is nearer, which chunked
detection reproduces exactly. `spacing="scipy"` follows `scipy.signal.find_peaks`.

Then [`deduplicate`](/dsp-kitchen/plugins/synapse/reference/python/detection/#deduplicate) keeps one
spike per event seen on several channels, and
[`snippets`](/dsp-kitchen/plugins/synapse/reference/python/extraction/#snippets) cuts the waveforms.

Worked examples: [Course: Detection](/dsp-kitchen/plugins/synapse/learn/course/detection/threshold-and-scoring/).
