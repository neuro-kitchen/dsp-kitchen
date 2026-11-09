---
title: Recordings
description: "Open, read and write recordings from Python."
sidebar:
  order: 1
---

Opening a file reads only its metadata. Samples are read when you ask for them.

```python
import dspk

rec = dspk.io.open("session.zarr")
rec.channel_count, rec.samples, rec.sample_rate, rec.duration
rec.channel_names, rec.units

first_second = rec.read(start=0, stop=30_000)              # all channels
some = rec.read(channels=[3, 0], start=0, stop=30_000)     # in the order given
```

`read` returns `(channels, samples)` float32 in each channel's unit.

## Files with several sources

Containers such as NWB hold several recordings. List them first:

```python
for s in dspk.io.sources("session.nwb.zarr"):
    print(s.id, s.channel_count, s.sample_rate)

rec = dspk.io.open("session.nwb.zarr", source="acquisition/EMG")
```

## Writing

```python
dspk.io.write_zarr(data, "out.zarr", 30_000.0, unit="µV")              # float32
dspk.io.write_raw(counts, "out.bin", 30_000.0, gain=0.195, unit="µV")  # integers kept as they are
```

For data that doesn't fit in memory, [`ZarrWriter`](/dsp-kitchen/reference/python/io/#zarrwriter)
takes blocks one at a time.

`dspk.io.formats()` lists what `open` recognizes. Neural formats (SpikeGLX, NWB) come with the
[Synapse plugin](/dsp-kitchen/plugins/synapse/learn/formats-and-probes/).

Full details: [`dspk.io`](/dsp-kitchen/reference/python/io/).
