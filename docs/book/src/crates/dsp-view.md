# dsp-view

## Intent

Preparing continuous multi-channel signals for viewing, wherever the viewer runs: an app next to
the recording, or a remote viewer served over the network by dsp-stream. A viewer draws
`[min, max]` per pixel column, which keeps every spike peak visible at any zoom; dsp-view
produces those columns without reading more samples than it must.

### Owns
- The min/max reduction on the host (vectorized, parallel) and on a CubeCL device.
- The min/max pyramid of a whole recording (in memory or in a file) and its background builder.
- `View` (what a viewer asks for) and `Envelope` (what it gets).
- The viewer's backend: `SignalBackend` (what a viewer may ask of a signal) and `LocalSignal`
  (a recording and its pyramid in this process). A remote session (dsp-stream) answers the same
  views; a viewer such as dsp-app only asks and draws.

### Must not contain
- Rendering or UI (dsp-app), network transport (dsp-stream), file formats (dsp-io).
- A dependency on dsp-base: dsp-view depends on dsp-core and CubeCL only.

## Features

| Feature | Default | Enables |
|---|---|---|
| `device` | via `wgpu` | `envelope_on_device` (CubeCL) |
| `wgpu` / `cpu` / `cuda` / `hip` | `wgpu` | the CubeCL runtime (forwarded to dsp-core) |

Without `device`, dsp-view runs on the host only.

## Module map

```text
dsp-view/src/
├── envelope/
│   ├── fold.rs      the min/max fold: columns (Columns::Even / Buckets), blocks in either memory order
│   ├── buckets.rs   min/max of one slice into N buckets
│   └── device.rs    envelope_kernel, envelope_on_device (feature device)
├── pyramid/
│   ├── mod.rs       Pyramid: fill (any order, concurrent), envelope, covers
│   ├── layout.rs    levels and pages
│   ├── storage.rs   memory or file bytes; PyramidIdentity, pyramid_path
│   └── builder.rs   PyramidBuilder: fills nearest the view first
├── signal.rs        SignalBackend, LocalSignal (pyramid policy, background builder), FILE_PYRAMID_MIN_BYTES
├── view.rs          View, Envelope, read / read_into (streamed raw reads)
└── read.rs          pipelined block reading (private)
```

## Usage

A viewer holds a `SignalBackend` and asks it for views:

```rust,ignore
use std::sync::Arc;
use dsp_view::{Envelope, LocalSignal, SignalBackend, View};

let signal: Arc<dyn SignalBackend> = Arc::new(LocalSignal::open(rec, Some((path, "main")))?);
signal.watch(Arc::new(|p| { /* more is built: redraw views that were incomplete */ }));
signal.focus(view.start);                       // build the pyramid here first
let mut out = Envelope::Samples(Vec::new());    // reused frame to frame
signal.view(&view, &mut out)?;
```

`LocalSignal::open` picks the pyramid: the complete file next to the recording if there is one;
else a new file there for recordings of at least `FILE_PYRAMID_MIN_BYTES` (64 MiB), filled once,
nearest the view first, while views already draw from it; else memory. `prepare(start, end)`
builds one range now (an exact single image, e.g. a snapshot).

The parts, used directly:

```rust,ignore
use std::sync::Arc;
use dsp_core::RecordingSource;
use dsp_view::{Envelope, Pyramid, PyramidBuilder, View};

let rec: Arc<dyn RecordingSource> = Arc::from(dsp_io::open(path)?);
let view = View { channels: (0..384).collect(), start: 0, end: rec.info().samples, width: 1920 };

// The file next to the recording if it is complete, else a new one (or memory)
let (pyramid, complete) = Pyramid::open_or_create(rec.as_ref(), Some((path, "main")))?;
let pyramid = Arc::new(pyramid);
if !complete {
    PyramidBuilder::default().run(rec.clone(), pyramid.clone(), view.start, cancel, on_progress);
}

match view.read(rec.as_ref(), Some(&pyramid))? {
    Envelope::Samples(s) => { /* fewer samples than columns: draw lines */ }
    Envelope::Columns { values, complete } => { /* draw [min, max]; ask again if !complete */ }
}
```

Data already on the device (for example after a processing pipeline):

```rust,ignore
let columns = view.columns();
let env = dsp_view::envelope_on_device::<f32>(&client, &handle, channels, samples, first, columns)?;
```

## Reference

| Item | What |
|---|---|
| `Columns::Even { start, len, width }` | `width` columns splitting `start..start + len`. |
| `Columns::Buckets { origin, size, count }` | `count` buckets of `size` samples from `origin`. |
| `fold_block`, `fold_row` | Fold samples into columns (channel-major or time-major; NaN skipped). |
| `min_max_decimate(_into)` | `[min, max]` of one slice in N buckets. |
| `envelope_on_device` | The same columns on a CubeCL device; one cube per `(channel, column)`; only columns are downloaded. |
| `Pyramid::in_memory(source, MEMORY_BASE)` | Empty pyramid in memory (`MEMORY_BASE = 256` samples per finest bucket, ≈ 1/64 of the recording as f32 once filled). |
| `Pyramid::{open_file, create_file, open_or_create}` | Pyramid in `<dir>/cache/<file>/<source>.minmax` (`FILE_BASE = 64`, ≈ 1/16), valid while the recording's size, modification time, shape and rate match. |
| `Pyramid::fill(source, start, end, block_values, stop)` | Build the pages (≈ `PAGE_SEC` = 1 s) holding `start..end`; any order, from several threads, each page once. |
| `Pyramid::envelope` | `[min, max]` per column from the coarsest level with ≤ 1 bucket per column, edges aligned to buckets (no peak lost); `false` when columns are finer than the base. |
| `PyramidBuilder::run` | Fill in the background, nearest the focus first; a new focus re-orders what is left. |
| `View::read(source, pyramid)` / `read_into(…, &mut Envelope)` | Samples, pyramid columns, or raw columns, whichever is exact and cheapest; `read_into` reuses the output buffer. Raw windows are streamed in blocks of at most `RAW_BLOCK_VALUES` values aligned to the source's storage chunks, the next block read while the current one folds: bounded memory, each chunk decoded once. |
| `SignalBackend` | `info`, `view`, `read` (samples, e.g. under the cursor), `focus`, `prepare`, `progress`, `watch`. |
| `LocalSignal::open(source, recording)` | The local backend: source, pyramid (policy above), `PyramidBuilder`; cancels the build when dropped. |

## Design rules

1. Exact envelopes: every sample falls into exactly one column; NaN means "unknown" or "all NaN",
   never a made-up value.
2. Read as little as possible: the pyramid serves zoomed-out views; raw reads only below its base.
3. One implementation per idea: one fold (host), one kernel (device), one pyramid (memory or file).

## Limitations

- Pyramid pages are not aligned to the recording's storage chunks, so a compressed store may
  decode a chunk twice at the edge of a read.
- On Windows an in-memory pyramid counts against the commit limit before it is filled.
- `envelope_on_device` uses 32-bit indices: at most 2³² values per buffer.
