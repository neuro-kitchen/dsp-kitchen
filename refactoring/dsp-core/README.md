# refactoring/dsp-core

Items removed from `crates/dsp-core` because they do not fit its intent:

> **dsp-core** — domain-agnostic building blocks for sampled multi-channel signals: exact time,
> buffers, errors, channel masks, chunk scheduling, the recording-source abstraction, and
> compute-runtime selection. No neural/probe/ML concepts.

- Removed on: 2026-10-05
- Branch / commit at removal: `versions/v0.14` @ `0f7e600`
- Build state: **intentionally broken** in downstream crates until the final fix-up pass.

---

## Parked files and folders

| Parked path | Original path | Reason | Candidate destination |
|---|---|---|---|
| ~~`layout/`~~ → **moved to `crates/dsp-io/src/neuro/probe/`** (2026-10-05, dsp-io phase 1) | `crates/dsp-core/src/layout/` | Probe geometry (shanks, µm positions, Neuropixels 1.0 preset, Phy/Kilosort channel arrays) is neuro domain. | Decided: `dsp-io/neuro/probe` (shared by io neuro formats and synapse). Aliases `ProbeLayout`, `ContactPosition`, `ChannelContact` dropped there. |
| `probe.rs` | `crates/dsp-core/src/probe.rs` | Backward-compatibility facade re-exporting `layout`; no external users of `dsp_core::probe`. | Delete. |

`layout/mod.rs` also defined the aliases `ProbeLayout`, `ContactPosition`, `ChannelContact`;
they left with it.

## Lines removed from files that stayed

| File | Removed | Reason |
|---|---|---|
| `src/lib.rs` | `pub mod layout;`, `pub mod probe;`, `pub use layout::{SensorLayout, SensorSite, Position3D, ProbeLayout};` | Modules parked. |
| `src/lib.rs` | `BufferChunk`, `MemoryLayout` from `pub use buffer::{…}` (replaced by `BufferLayout`) | Aliases removed. |
| `src/buffer/mod.rs` | `pub type BufferChunk = SignalChunk;` / `pub type MemoryLayout = BufferLayout;` | Aliases with no external users. |
| `src/error.rs` | `#[error("Model inference failed: {0}")] Model(String),` | ML concern; `dsp-synapse-ml` should own its error type. |
| `src/time/rate.rs` | `pub const STANDARD_AUDIO: f64 = 48_000.0;` | Unused. |
| `src/recording/info.rs` | `use crate::layout::SensorLayout;`, field `pub layout: Option<SensorLayout>`, init `layout: None` | Geometry no longer attached to recordings in core. |
| `src/recording/slice.rs` | `info.layout = p_info.layout.as_ref().map(\|l\| l.select_channels(&channel_map));` | Follows `RecordingInfo.layout` removal. The neuro layer must re-apply `select_channels` when slicing channels. |

## Downstream breakage (to fix at the end)

Snapshot of references to removed items at removal time (files per crate):

- **`SensorLayout` / `SensorSite` / `Position3D` / `dsp_core::layout`**
  - `dsp-synapse` — ~20 files (`probe/`, `spatial/`, `detection/dedup.rs`, `extraction/`,
    `storage/`, `streaming/`, `core/`, tests)
  - `dsp-io` — `spikeglx.rs` (builds and attaches a layout)
  - `dsp-cli` — `commands/benchmark.rs`, `commands/components.rs`, `commands/probe.rs` (`get_contact`)
  - `dsp-app` — `engine/data/dataset.rs`, `views/explore.rs`, `viewmodels/explore.rs`, `app.rs`
  - `dsp_kitchen_py` — `synapse/probe.rs`
- **`ProbeLayout` alias** — `dsp-synapse/src/streaming/runner.rs`; `dsp_kitchen_py` (`lib.rs`, `synapse/*`)
- **`RecordingInfo.layout`** — `dsp-io/spikeglx.rs`, `dsp-cli/commands/info.rs`, `dsp-app` (dataset, explore)
- **`DspError::Model`** — `dsp-synapse-ml` (`models/*`, `runtime/*`, ~65 sites)

## API changes in dsp-core (not parked, but break downstream)

| Before | After | Downstream files using the old form |
|---|---|---|
| `ChannelInfo { name, gain_uv, offset_uv }` | `ChannelInfo { name, gain, offset, unit: SignalUnit }` | `gain_uv` 15, `offset_uv` 9 |
| `RecordingInfo::with_gain_uv(g)` | `RecordingInfo::with_gain(g, unit)` | 3 |
| `RecordingInfo.start_time_sec: f64` | `RecordingInfo.start_time: RationalTime` (use `RationalTime::from_seconds_f64` for decimal inputs) | 8 |
| `SampleRate { rate_hz: f64 }` (serde: `{"rate_hz": f64}`) | `SampleRate { hz: Ratio<u64> }` (serde: `{"hz": [n, d]}`); `new(f64)` unchanged, added `from_ratio(n, d)` | serde shape only; no persisted users found |
| reads documented as µV | reads return values in each channel's `SignalUnit`; default `Dimensionless` | readers that set gains must also set `unit` (e.g. `Microvolt`) |

New: `recording::SignalUnit` (`Volt`, `Millivolt`, `Microvolt`, `Ampere`, `Milliampere`,
`Microampere`, `Dimensionless` (default), `Other(String)`), `RationalTime::from_seconds_f64`,
`RationalTime::checked_add`.

## 2026-10-06 — additions for dsp-stream

- `RationalTime::as_ratio()`: the exact fraction of seconds (mirrors `SampleRate::as_ratio`), so
  a stream header carries the start time exactly.
- `SampleFormat::decode(bytes, out, gain, offset)`: little-endian stored words → scaled `f32`,
  every format, with a test. **One decoder:** dsp-io's private `container/binary/codec.rs::decode_run`
  is the same code and should call this instead (dsp-io fix-up pass).

## 2026-10-06 — orchestration moved into core

`window.rs` → `window/{schedule.rs, loader.rs}` ("Orchestration" in the book). `WindowLoader`
(from dsp-io's `PrefetchReader` and the short-lived `dsp-orchestrate` crate, both removed) streams
any `&[HaloWindow]` with a background read-ahead: `stream`, `stream_stored`, `stream_while` (stops
when `f` returns `false`). `HaloWindow::around(index, valid, left, right, total)` and
`HaloWindow::remap_event(local)` replace hand-built windows and the separate remap helpers.
**Why core:** it needs only `RecordingSource`, `ChunkSchedule` and `HaloWindow` (all here) and
std threads; dsp-base and dsp-view depend only on dsp-core, so they get out-of-core reading
without dsp-io. Breaking: `dsp_io::PrefetchReader` and `dsp_orchestrate::*` are gone
(`for_each_window(f)` → `WindowLoader::new(src).stream(schedule.windows(), f)`).
