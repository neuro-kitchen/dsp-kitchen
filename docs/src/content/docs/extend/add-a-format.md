---
title: Add a format
description: "Teach dspk-io to read another file format."
sidebar:
  order: 2
---

A format is a unit struct implementing `dspk_io::core::Format`, listed in the registry.

```rust
pub trait Format: Send + Sync {
    /// Stable name ("raw", "zarr-traces", "nwb", ...).
    fn name(&self) -> &'static str;
    /// Whether `path` is in this format, from its path and metadata only.
    fn detect(&self, path: &Path) -> bool;
    /// The signals `path` offers, from metadata only.
    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>>;
    /// Opens source `id`.
    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>>;
}
```

`detect` and `sources` must never read sample data: they run on every file a user points at.

## The reader

`open` returns a `dspk_core::RecordingSource`:

- `info()` describes the recording.
- `read(channels, samples, out)` fills `out`, channel-major, scaled to each channel's unit.
- `read_stored(channels, samples, out)` fills `out` with the stored values, before gain and offset.

Implement `read_stored` whenever the file holds integers. It's what lets pipelines
[move compact samples](/dsp-kitchen/gpu/data-movement/) and convert them on the device.

## Where it goes

| Kind | Crate |
|---|---|
| A generic format | `dspk-io`, listed in its registry |
| A neural format | `dspk-synapse`'s `io`, listed in its `formats()` |

Order matters in the registry: formats are tried in order, so a more specific format goes first.

## Test

Round-trip through a small file on disk, check `detect` refuses other formats, and compare
`read_stored` with `read` through the gain and offset.
