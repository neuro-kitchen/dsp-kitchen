# dsp-io

> **Status:** restructured on 2026-10-05 (`versions/v0.14`): phase 1 (layered tree), phase 1b
> (format registry) and phase 2 (shared containers) done; phase 3 (sorting formats) pending with
> the dsp-synapse review. Not yet compiled; downstream crates are intentionally broken until the
> final fix-up pass. Moves are tracked in `refactoring/dsp-io/README.md`.

## Intent

Reading (and writing) recording files. Every format turns a file into a
[`dsp_core::RecordingSource`](dsp-core.md#recording), so nothing above this crate depends on how
data is stored.

### Owns
- **Containers**: how bytes are stored (binary codecs, `.npy`, Zarr v3), one implementation each
  for the whole workspace.
- **Format schemas**: conventions on top of a container (raw + sidecar, Zarr `/traces`, NWB,
  SpikeGLX, mtscomp), and detection of which one a path is.
- **Neural file metadata** (feature `neuro`): probe geometry as stored by acquisition formats.

### Must not contain
- Algorithms (filtering, detection, sorting): those consume `RecordingSource`.
- Domain models of other crates (e.g. synapse's `SortingOutput`). Sorting formats return data
  shaped like the file; conversion to domain types lives in the domain crate.
- Format names inside `core/` or `container/`: only `registry.rs` lists formats.

## Layers

```text
registry.rs ──lists──▶ generic/*, neuro/*  ──implement──▶ core::Format
                              │
                              └──use──▶ container/*  (binary, npy, zarr)
core/  ──defines──▶ Format, SourceEntry; open() loops over registry::formats()
```

Dependencies point one way: `container` knows no formats, `core` knows only the trait, formats
know `core` + `container`, and `registry.rs` is the single place that knows every format.

## Features

| Feature | Default | Enables |
|---|---|---|
| `zarr` | yes | `container::zarr`, `generic::zarr_traces`, and (with `neuro`) NWB. |
| `neuro` | **no** | `neuro::*`: NWB, SpikeGLX, mtscomp, probe geometry, synthetic recordings. |

A user of `dsp-base` alone gets raw binary and Zarr `/traces` with no neural code.

## Module map

```text
dsp-io/src/
├── lib.rs                   root re-exports
├── registry.rs              formats(): every Format, in detection order + feature gates
├── core/
│   ├── format.rs            Format trait
│   ├── open.rs              detect, open, sources, open_source
│   ├── sources.rs           SourceEntry, SourceKind, MAIN, single_source, require_main, default_source
│   └── cached.rs            CachedRecording
├── container/
│   ├── binary/codec.rs      little-endian decode/encode, frame scaling (crate-internal)
│   ├── npy/                 .npy read/write
│   └── zarr/                Zarr v3 store, groups, arrays            (feature zarr)
├── generic/
│   ├── raw/                 RawRecording, RawParams, write_raw, Raw
│   └── zarr_traces/         ZarrRecording, write_zarr, ZarrTraces    (feature zarr)
└── neuro/                                                            (feature neuro)
    ├── nwb/                 acquisition.rs (NwbZarrRecording), units.rs, Nwb   (+ zarr)
    ├── spikeglx/            SpikeGlxMeta, apply_meta, probe_layout, SpikeGlx
    ├── mtscomp/             MtscompRecording, Mtscomp
    ├── probe/               SensorLayout, SensorSite, Position3D, ProbeSource, probe_of
    └── synthetic/           SyntheticRecording, SyntheticParams
```

Each format folder has the same shape: `mod.rs` (the reader, implementing `RecordingSource`)
and `format.rs` (a unit struct implementing `Format`). See
[Adding a format](../guides/adding-a-format.md).

## Reference

### `core`
| Item | Purpose |
|---|---|
| `Format` | `name`, `detect` (path + metadata only), `sources`, `open(path, id)`, `open_default` (provided: `sources` + `default_source`; single-source formats override it). |
| `detect(path)` | First registered `Format` whose `detect` accepts `path`; one error listing compiled-in formats otherwise. |
| `open(path)` | `detect(path)?.open_default(path)`. |
| `sources(path)` / `open_source(path, id)` | List signals of a file / open one by id. |
| `SourceEntry` | `id`, `name`, `kind`, `channels`, `samples`, `sample_rate: SampleRate`, `start_time: RationalTime`, `format: SampleFormat`, `unit: SignalUnit`; `duration_sec`, `summary`. |
| `SourceKind` | `Electrical` (volt units) / `Other`; `from_unit`. |
| `MAIN`, `single_source`, `require_main` | Helpers for single-recording formats. |
| `default_source` | Largest electrical source, else largest. |
| `CachedRecording` | Wraps a chunked source; keeps recently decoded chunks within a memory budget. |

### `registry`
| Item | Purpose |
|---|---|
| `formats()` | `&'static [&'static dyn Format]` in detection order: `nwb`, `zarr-traces`, `spikeglx`, `mtscomp`, `raw`. Order matters: NWB stores are also Zarr stores; SpikeGLX `.cbin` files are also mtscomp. |

### `container`
| Module | Contents |
|---|---|
| `binary::codec` | Crate-internal: decode/encode little-endian runs with gain/offset, select stored channels, native↔LE. |
| `npy` | `read_npy::<T>` (v1–3, all numeric dtypes, both byte orders, C/Fortran → C order, exact integer conversion), `write_npy`, `read_header`, typed 1-D/2-D/3-D helpers, `NpyArray`, `NpyElement`, `NpyWritable`. |
| `zarr` | `open_store`, `open_rw_store`, `write_group`, `read_node_json`, `read_node_attributes`, `has_array`, `write_array_{u64,i64,i32,f32,f64}` (hdmf-zarr `_DTYPE` / `_ARRAY_DIMENSIONS`), `read_array::<T>` (falls back to legacy `<node>.npy`), `read_optional_array`, `read_all`. |

### `generic`
| Format | Files | Notes |
|---|---|---|
| `raw` (`Raw`) | `rec.bin` + JSON sidecar `rec.meta` | Memory-mapped. Sidecar: `channels`, `sample_rate_hz`, `format`, `order`, `gain_uv`, `offset_uv`, `header_bytes`, optional `samples`. `write_raw` writes data + sidecar. |
| `zarr-traces` (`ZarrTraces`) | Zarr v3 store with `/traces` | `[channels, samples]` or `[samples, channels]` (by dimension names); root attrs `sample_rate_hz`. `write_zarr`, `DEFAULT_CHUNK_SAMPLES`. |

### `neuro` (feature `neuro`)
| Format | Files | Sources | Notes |
|---|---|---|---|
| `nwb` (`Nwb`) | `.nwb.zarr` (hdmf-zarr) | one per continuous `/acquisition` series | Default: largest `ElectricalSeries`. Electrical / volt series read in µV. Channel names from the electrodes table. `units.rs`: `infer_nwb_sample_rate`. |
| `spikeglx` (`SpikeGlx`) | `.bin` / `.cbin` + `.meta` | one per stream of the run (`imecN.ap`, `imecN.lf`, `nidq`) | int16 interleaved; µV gains from `imroTbl` / `niMNGain` / `niMAGain`. Implements `ProbeSource`. |
| `mtscomp` (`Mtscomp`) | `.cbin` + `.ch` | one | IBL compression; chunks decoded in parallel and kept for the next read. |

| Item | Purpose |
|---|---|
| `probe::SensorLayout` / `SensorSite` / `Position3D` | Probe geometry (µm positions, shanks); `select_channels`, `to_channel_arrays` / `from_channel_arrays` (Phy/Kilosort arrays), `neuropixels_1_0_standard`. |
| `probe::ProbeSource` | Optional capability of a `Format`: `probe(path, id) -> Option<SensorLayout>`. |
| `probe::probe_of(path, id)` | Geometry of a source, kept **beside** the recording (core's `RecordingInfo` carries none). Apply `select_channels` when slicing channels. |
| `spikeglx::probe_layout(meta)` | Geometry from `snsGeomMap`, else `snsShankMap` + probe pitch. |
| `synthetic::SyntheticRecording` | Deterministic procedural recording (noise, hum, drifting spike trains) with ground truth, for tests and demos. |

## Design rules

1. **One container implementation per storage technology** for the whole workspace.
2. **`detect` never reads samples**: path and metadata only.
3. **Geometry beside, not inside**: probe layouts come from `probe_of`, not from `RecordingInfo`.
4. **Formats return file-shaped data**; domain crates convert to their own types.
5. **Feature gates live in `registry.rs`** (and the module declarations they mirror).

## Consumers

`dsp-app`, `dsp-cli`, `dsp-synapse-ml`; `dsp-synapse` after the fix-up pass (it currently uses
`zarrs` directly).

## Open items

- **Phase 3**: Phy (`phy.rs`, `phy_sorting.rs`), NWB `/units` (`nwb_units.rs`) and
  `.sorting.zarr` (`zarr_analyzer.rs`) from `dsp-synapse/storage` → `neuro/phy/`,
  `neuro/nwb/units.rs`, `neuro/sorting_zarr/`, split from synapse's `SortingOutput`.
- NWB does not yet implement `ProbeSource` (the electrodes table holds positions).
- `core/` tests still use `SyntheticRecording` and so only run with `neuro`.
