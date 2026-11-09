---
title: Formats and probes
description: "Open neural recordings and describe the probe."
sidebar:
  order: 1
---

## Recordings

On top of the core's raw binary and Zarr, Synapse reads:

| Format | Files |
|---|---|
| SpikeGLX | `.bin` or compressed `.cbin`, with its `.meta` |
| mtscomp | Compressed electrophysiology files |
| NWB | `.nwb.zarr` stores; each electrical series is a source |

```sh
dspk open session.imec0.ap.bin       # its layout and read speed
```

## Probes

A probe is the position of the site under every recording channel, in µm, with `y` as depth.

```python
import dspk

probe = dspk.synapse.io.Probe.preset("neuropixels1")    # also neuropixels2, tetrode, utah, hdemg4x8, hdemg8x8
probe = dspk.synapse.io.Probe.grid(8, 8, pitch_um=8000.0)
probe = dspk.synapse.io.probe_of("session.imec0.ap.bin")   # what the file carries (SpikeGLX), or None
```

A probe file (`.json`: probeinterface or Kilosort 4) also says which channel each contact is wired to:

```python
probe, map = dspk.synapse.io.read_probe_file("probe.json")
```

`.mat` probe files are refused. A probe follows a remap of its recording with `probe.remap(map)`, and
`probe.depth_order()` gives the map that sorts channels by depth. On the command line, `dspk probe`
shows a layout and `--write-map` writes the map.

## Sortings

A [`Sorting`](/dsp-kitchen/plugins/synapse/reference/python/io/#sorting) holds units with their spike
trains, templates and quality metrics.

| Format | How it's chosen when saving |
|---|---|
| NWB `/units`, in a `.nwb.zarr` | Path ends in `.nwb.zarr` |
| `.sorting.zarr` | Any other `.zarr` path |
| Phy / Kilosort folder | Anything else |

`dspk convert in out` converts between them.
