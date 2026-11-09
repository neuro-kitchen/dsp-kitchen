---
title: Architecture
description: "The crates, what depends on what, and the rules that keep them apart."
sidebar:
  order: 1
---

```text
dspk-core        recordings, buffers, time, windows, compute targets
   ├─ dspk-io        files: raw binary, Zarr, detection, chunk cache
   ├─ dspk-ops       filters, resampling, spatial, linalg, statistics, peaks, device pipelines
   │     └─ dspk-pipeline   pipelines as data: specs, plans, many inputs, provenance
   ├─ dspk-view      min/max envelopes and pyramids
   │     └─ dspk-stream     QUIC + TLS sessions, the server's protocol
   └─ dspk-hub       verified downloads

dspk-synapse     electrophysiology on top: formats, probes, detection, clustering, metrics
   ├─ dspk-synapse-ml         pretrained models
   └─ dspk-sorter-{kilosort, ms5, si}     one crate per upstream project

dspk-cli, dspk-app (+ dspk-app-core, dspk-app-synapse), the Python package
```

## Folders

| Folder | Holds |
|---|---|
| `crates/` | The core: `dspk`, `dspk-core`, `dspk-io`, `dspk-ops` |
| `engine/` | `dspk-pipeline` |
| `serve/` | `dspk-view`, `dspk-stream` |
| `synapse/`, `hub/` | The electrophysiology plugin and downloads |
| `app/`, `plugins/` | Command line, app, and their plugins |
| `bindings/python/` | The Python package |
| `playground/`, `benchmarks/`, `plating/` | The course, the measurements, and their shared plots |

## Rules that keep it together

- **One kernel source.** A kernel is written once, in CubeCL, and runs everywhere. Host versions exist
  only as test references.
- **A window crosses each way at most once.** Stages stay on the device between upload and download.
- **Stored type to the device.** Recordings aren't converted on the host.
- **Viewing lives in one crate.** `dspk-view` holds it, whether for the app or a remote viewer.
- **The app only draws.** Analysis comes from `dspk-synapse` and `dspk-ops`.
- **Plugins are compiled in.** The core handles any multi-channel signal; plugins bring a domain.
- **Licences stay separated.** Nothing MIT or Apache-licensed links `dspk-sorter-kilosort`: it reaches
  users through the separate `dspk-cli-sorter` program.
- **Every setting has a visible default**: a config field, or a Python keyword with its default in the
  signature.
