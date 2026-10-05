# refactoring/dsp-io

Restructure of `crates/dsp-io` from a flat layout into layers.

- Date: 2026-10-05
- Branch / commit before restructure: `versions/v0.14` @ `0f7e600`
- Build state: **intentionally broken** until the final fix-up pass.

Nothing from dsp-io is parked here yet; this file tracks moves and path changes.

## Phase 1 — file moves (done)

| Before | After |
|---|---|
| `src/lib.rs` (`open` + tests) | `src/core/open.rs` (test helpers now `crate::core::tests`) |
| `src/sources.rs` | `src/core/sources.rs` |
| `src/cached.rs` | `src/core/cached.rs` |
| `src/codec.rs` | `src/container/binary/codec.rs` |
| `src/raw.rs` | `src/generic/raw/mod.rs` |
| `src/zarr.rs` | `src/generic/zarr_traces/mod.rs` (store helpers to `container/zarr/` in phase 2) |
| `src/nwb.rs` | `src/neuro/nwb/acquisition.rs` |
| `src/spikeglx.rs` | `src/neuro/spikeglx/mod.rs` |
| `src/mtscomp.rs` | `src/neuro/mtscomp/mod.rs` |
| `src/synthetic.rs` | `src/neuro/synthetic/mod.rs` |
| `refactoring/dsp-core/layout/` | `src/neuro/probe/` (aliases `ProbeLayout`, `ContactPosition`, `ChannelContact` removed) |

Root re-exports kept (neuro ones only with `neuro`): `open`, `sources`, `open_source`, `default_source`, `SourceEntry`,
`SourceKind`, `CachedRecording`, `RawParams`, `RawRecording`, `write_raw`, `ZarrRecording`,
`write_zarr`, `NwbZarrRecording`, `MtscompRecording`, `SpikeGlxMeta`, `SyntheticParams`,
`SyntheticRecording`; new: `SensorLayout`, `SensorSite`, `Position3D`.

### Downstream module paths that changed (fix at the end)

| Old path | New path | Used in |
|---|---|---|
| `dsp_io::nwb::…` | `dsp_io::neuro::nwb::…` | 1 site |
| `dsp_io::raw::…` | `dsp_io::generic::raw::…` | 1 site |
| `dsp_io::synthetic::…` | `dsp_io::neuro::synthetic::…` | 1 site |
| `dsp_io::zarr::DEFAULT_CHUNK_SAMPLES` | `dsp_io::generic::zarr_traces::DEFAULT_CHUNK_SAMPLES` | 1 site |
| `dsp_io::sources::MAIN` | unchanged (also `dsp_io::core::MAIN`) | — |
| `dsp_core::layout::*` / `ProbeLayout` | `dsp_io::{SensorLayout, SensorSite, Position3D}` | see `refactoring/dsp-core/README.md` |

### Known breakage inside dsp-io
- All readers still use `gain_uv` / `offset_uv` / `start_time_sec` / `with_gain_uv` (core API
  change). When fixing: raw sidecars (`gain_uv`), SpikeGLX and NWB electrical series must set
  `unit = SignalUnit::Microvolt`.

## Phase 1b — format registry (done)

- `core/format.rs`: `Format` trait (`name`, `detect`, `sources`, `open`, `open_default`).
- `src/registry.rs`: static `formats()` list in detection order (nwb, zarr-traces, spikeglx,
  mtscomp, raw); the only place holding format feature gates.
- Each format folder has a `format.rs` with its unit struct: `Raw`, `ZarrTraces`, `Nwb`,
  `SpikeGlx`, `Mtscomp`.
- `core/open.rs`: `detect`, `open`, `sources`, `open_source` loop over `formats()`; `core/`
  no longer names any format (only `neuro`-gated tests use `SyntheticRecording`).
- SpikeGLX stream discovery moved from `core/sources.rs` to `neuro/spikeglx/format.rs`.
- `neuro/probe/source.rs`: `ProbeSource` trait + `probe_of(path, id)`; SpikeGLX implements it via
  new `spikeglx::probe_layout(meta)` (geometry removed from `apply_meta`).
- Cargo feature `neuro` added, **off by default** (`default = ["zarr"]`).

### Downstream changes from phase 1b (fix at the end)

| Change | Action for consumers |
|---|---|
| `neuro` feature off by default | `dsp-app`, `dsp-cli`, `dsp-synapse(-ml)`, `dsp_kitchen_py`: enable `features = ["neuro"]` where they use NWB / SpikeGLX / mtscomp / probe / synthetic. |
| `RecordingInfo.layout` from SpikeGLX gone | Call `dsp_io::probe_of(path, id)`; apply `select_channels` when slicing channels. Users: `dsp-app` (`engine/data/dataset.rs`), `dsp-cli` (`commands/info.rs`). |
| `SourceEntry.sample_rate: f64` | now `SampleRate` (`.rate_hz()`) |
| `SourceEntry.start_time_sec: f64` | now `start_time: RationalTime` |
| `SourceEntry.unit: String` | now `SignalUnit` (`.symbol()` for display) |
| new `SourceEntry.format: SampleFormat` | — |
| `nwb::SeriesEntry` new field `format`; series with unreadable dtypes are no longer listed | — |
| `open()` error for unknown paths is now one generic message listing compiled-in formats | — |

## Phase 2 — container layer (done)

| Change | Detail |
|---|---|
| `container/npy/` | from `dsp-synapse/src/storage/npy.rs` (the complete reader/writer: v1–3, all numeric dtypes, both byte orders, C/Fortran). No dependencies; always compiled. |
| `container/zarr/` | from `dsp-synapse/src/storage/zarr_store.rs` (feature `zarr`). Added `open_store` (no mkdir; `open_rw_store` now calls it) and `read_all` (from `neuro/nwb`). |
| `generic/zarr_traces` | private `open_store` deleted; uses `container::zarr::open_store`. |
| `neuro/nwb/acquisition.rs` | private `node_meta` and `read_all` deleted; uses `container::zarr::{read_node_json, read_all}`. |
| `neuro/nwb/units.rs` | new; holds `infer_nwb_sample_rate` (was in synapse's `zarr_store`). |
| synapse-ml `hub/npy.rs` | parked in `refactoring/dsp-synapse-ml/`. |

Remaining duplicates: none of `.npy`; Zarr store-opening now has one implementation.
## Addition — out-of-core reader (2026-10-05)

`PrefetchReader` moved here from dsp-stream (`src/core/prefetch.rs`, `dsp_io::PrefetchReader`): it reads
local recording windows on a background thread; it never belonged to the network crate.

## Addition — probe presets and neighbours (2026-10-05, synapse step 1a)

- `src/neuro/probe/presets.rs`: `neuropixels_1_0` (moved out of `SensorLayout::neuropixels_1_0_standard`,
  method removed), `neuropixels_2_0`, `hdemg_grid` / `hdemg_4x8` / `hdemg_8x8`, `tetrode`,
  `utah_array` (from dsp-synapse `probe/`; nominal geometry, spacings named).
- `src/neuro/probe/neighbors.rs`: `find_k_nearest_neighbors`, `precompute_knn_table` (from
  dsp-synapse `probe/neighbors.rs`).

## Phase 3 — sorting formats (in progress, 2026-10-05)

Design agreed: dsp-io owns the sorting **files** (plain structs mirroring them, read / write,
detection); dsp-synapse converts them to / from `SortingOutput`.

- Done — Phy / Kilosort: `src/neuro/phy/mod.rs` (`PhyFolder`, `PhyParams`, `ClusterTables`,
  `resolve_dat_path`, `ClusterId`) and `src/neuro/templates.rs` (`DenseTemplates`,
  `TemplateAxisOrder`), from dsp-synapse `storage/phy_sorting.rs`, `storage/phy.rs`,
  `core/template.rs`. New dependency: `tracing` (warning on unreadable optional arrays).
- Done — NWB `/units`: `src/neuro/nwb/units.rs` `NwbUnitsTable` (read / write; joins
  `infer_nwb_sample_rate`), from dsp-synapse `storage/nwb_units.rs`.
- Done — `.sorting.zarr` (dsp-kitchen's own format): `src/neuro/sorting_zarr/mod.rs`
  `SortingZarr` (+ manifest types), from dsp-synapse `storage/zarr_analyzer.rs` (feature `zarr`).
- Done — detection: `src/neuro/sorting_format.rs` `SortingFormat`, `detect_sorting`
  (zarr-based checks only with feature `zarr`).

### Earlier notes
