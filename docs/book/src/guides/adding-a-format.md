# Adding a format to dsp-io

A format is one folder plus one line in `registry.rs`. Nothing in `core/` or `container/`
changes.

## 1. Pick the layer

| Layer | When | Feature gate |
|---|---|---|
| `generic/` | Any domain could use it (plain binary, generic Zarr/HDF5 layouts, audio, …). | none, or the container's (`zarr`) |
| `neuro/` | It encodes neural-recording conventions (probes, electrodes, acquisition systems). | `neuro` (+ container's) |

If the format needs a storage technology dsp-io does not have yet (e.g. HDF5), add it under
`container/<name>/` first, behind its own feature, with no knowledge of any format.

## 2. Create the folder

```text
<layer>/<name>/
├── mod.rs       the reader: a struct implementing dsp_core::RecordingSource
└── format.rs    a unit struct implementing core::Format
```

Large formats split `mod.rs` by concern (e.g. `nwb/acquisition.rs`, `nwb/units.rs`). Create
version subfolders only when versions actually differ in code.

## 3. The reader (`mod.rs`)

```rust
//! <Name> recordings: <files>, <layout>.

use std::ops::Range;
use std::path::Path;

use dsp_core::recording::check_read;
use dsp_core::{DspResult, RecordingInfo, RecordingSource};

mod format;
pub use format::<Name>;

pub struct <Name>Recording {
    info: RecordingInfo,
    // handle to the data (mmap, zarr array, …)
}

impl <Name>Recording {
    pub fn open(path: &Path) -> DspResult<Self> { … }
}

impl RecordingSource for <Name>Recording {
    fn info(&self) -> &RecordingInfo { &self.info }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        // fill out[i * n..(i + 1) * n] with channels[i], scaled to each channel's unit
        …
    }
}
```

Rules:
- Set every `ChannelInfo`'s `gain`, `offset` **and `unit`** (`SignalUnit`); never assume µV.
- Set `RecordingInfo.start_time` (`RationalTime::from_seconds_f64` for decimal inputs).
- Implement `read_stored` when the stored type is compact (integers), `read_native` when the
  stored order is time-major, `chunk_samples` when the storage is chunked.
- Use `container::*` helpers (`binary::codec`, `npy`, `zarr`); never open a second copy of a
  container.

## 4. The format (`format.rs`)

Single-recording file:

```rust
//! [`Format`] for <Name> files.

use std::path::Path;

use dsp_core::{DspResult, RecordingSource};

use super::<Name>Recording;
use crate::core::sources::{require_main, single_source, SourceEntry};
use crate::core::Format;

pub struct <Name>;

impl Format for <Name> {
    fn name(&self) -> &'static str { "<name>" }

    /// Path and metadata only; never read samples here.
    fn detect(&self, path: &Path) -> bool { … }

    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>> {
        Ok(vec![single_source(&<Name>Recording::open(path)?)])
    }

    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
        require_main(path, id)?;
        self.open_default(path)
    }

    fn open_default(&self, path: &Path) -> DspResult<Box<dyn RecordingSource>> {
        Ok(Box::new(<Name>Recording::open(path)?))
    }
}
```

Multi-signal file (several series / streams): `sources` lists one `SourceEntry` per signal from
metadata only, with a stable `id`; `open` opens by that id; keep the provided `open_default` or
override it with a cheaper path.

## 5. Register it

In `src/registry.rs`, add one line at the right position (first match wins: put more specific
formats before more general ones that would also accept the path):

```rust
#[cfg(feature = "neuro")]
&crate::neuro::<name>::<Name>,
```

and declare the module in `<layer>/mod.rs` with the same gate.

## 6. Optional: probe geometry (`neuro` only)

If the file describes the probe, implement `neuro::probe::ProbeSource` for the format struct and
add it to `PROBE_SOURCES` in `neuro/probe/source.rs`.

## 7. Tests

- In `mod.rs`: write a tiny file in a temp dir, open it through `crate::open`, check `info()` and a
  `read`.
- Use `crate::core::tests::assert_stored_matches` / `assert_native_matches` when implementing
  `read_stored` / `read_native`.

## 8. Docs

Add a row to the format tables in [dsp-io](../crates/dsp-io.md).

## Checklist

- [ ] Folder `<layer>/<name>/` with `mod.rs` + `format.rs`
- [ ] `detect` reads no samples
- [ ] Gain, offset and `SignalUnit` set per channel; `start_time` set
- [ ] Only `container::*` used for storage
- [ ] One line in `registry.rs` at the right position, with feature gate
- [ ] `ProbeSource` if the file carries geometry
- [ ] Tests and docs row
