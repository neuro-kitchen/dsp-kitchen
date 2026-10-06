# dsp-core

## Intent

Domain-agnostic building blocks for sampled multi-channel signals: exact time, buffers, errors,
channel masks, chunk scheduling, the recording-source abstraction, and compute-runtime selection.

### Owns
- The shared vocabulary every other crate speaks (time, sample rate, buffers, errors).
- The `RecordingSource` contract: how any recording is described and read, independent of format.
- The single entry point to a CubeCL runtime (`compute` feature).

### Must not contain
- Neural, probe, or sorting concepts (probe geometry, spikes, units/clusters, Phy/Kilosort formats).
- ML concepts (model errors, inference).
- File-format implementations (those live in `dsp-io`).
- DSP algorithms or kernels (those take a `ComputeClient<R>` from here and live in algorithm crates).

## Features

| Feature | Enables |
|---|---|
| *(default)* | Types only; no CubeCL dependency. |
| `compute` | `compute` module (CubeCL). |
| `wgpu` / `cpu` / `cuda` / `hip` | `compute` + that runtime. |

`device.rs` (`ComputeTarget`) is always compiled, so a runtime can be named and validated without
GPU libraries.

## Module map

```text
dsp-core/src/
├── lib.rs           re-exports
├── error.rs         DspError, DspResult
├── time/
│   ├── rational.rs  RationalTime
│   ├── rate.rs      SampleRate
│   └── range.rs     TimeRange
├── buffer/
│   ├── layout.rs    BufferLayout, MemoryOrder
│   └── chunk.rs     SignalChunk
├── mask/
│   └── channel.rs   ChannelMask
├── window/          orchestration of recordings larger than memory
│   ├── schedule.rs  ChunkSchedule, HaloWindow
│   └── loader.rs    WindowLoader
├── recording/
│   ├── source.rs    RecordingSource, check_read, check_read_stored
│   ├── info.rs      RecordingInfo, ChannelInfo
│   ├── unit.rs      SignalUnit
│   ├── format.rs    SampleFormat
│   ├── memory.rs    MemoryRecording
│   └── slice.rs     SlicedRecording
├── device.rs        ComputeTarget, ComputeError, RUNTIME_ENV
└── compute/         (feature `compute`)
    ├── target.rs    ComputeTask, ComputeTarget::run
    ├── launch.rs    LaunchGeometry, sample_position, channel_position, row_position
    ├── tune.rs      tune_id, size_class
    └── bench.rs     sync, time_device
```

## Reference

### `time`
| Type | Purpose |
|---|---|
| `RationalTime` | Seconds as an exact `Ratio<u64>`. `from_samples`, `to_sample_index` (exact at fractional rates), `from_seconds_f64`, `checked_add`, `ZERO`. |
| `SampleRate` | Rate stored as an exact `Ratio<u64>` Hz. `new(f64)` keeps the exact decimal (`30000.12` → `750003/25`); `from_ratio(n, d)`; `rate_hz`, `nyquist_hz`, duration ↔ samples helpers. |
| `TimeRange` | `[start, end]` of `RationalTime`. Currently unused; kept deliberately. |

### `buffer`
| Type | Purpose |
|---|---|
| `MemoryOrder` | `ChannelMajor` (per-channel contiguous) or `TimeMajor` (interleaved). |
| `BufferLayout` | `channels × samples` + order; `linear_index(channel, sample)`. |
| `SignalChunk` | Owned host `f32` buffer with layout, sample rate and start time; bounds-checked `get`/`set`. |

### `mask`
| Type | Purpose |
|---|---|
| `ChannelMask` | Enabled/disabled flag per channel; `active_indices`, `active_count`. |

### `window`: orchestration (recordings larger than memory)

Use this module whenever a computation runs over a whole recording: never read a recording into
memory at once.

| Type | Purpose |
|---|---|
| `ChunkSchedule` | Splits a sample range into non-overlapping windows of `batch_samples`, each padded by a left/right halo clamped to the recording. `max_read_samples` sizes persistent buffers. |
| `HaloWindow` | One window: `valid_global` (samples it owns), `read_global` (samples it reads), `valid_local` (owned part of the read buffer). `around(index, valid, left, right, total)` builds one window; `remap_event(local)` maps an event in the read buffer to a recording sample only if this window owns it (no duplicates across windows). |
| `WindowLoader` | `new(source)` / `with_channels`. `stream(windows, f)` calls `f(&window, samples)` for every window of any list (a whole schedule, every n-th window, isolated windows) with the channel-major `[channels, read_len]` scaled values, reading the next window on a background thread; `stream_stored` gives the stored bytes; `stream_while` stops when `f` returns `false`. Host memory stays at two windows; no allocation per window. |

```rust,ignore
use dsp_core::{ChunkSchedule, WindowLoader};

let schedule = ChunkSchedule::full_recording(info.samples, batch, left_halo, right_halo);
WindowLoader::new(&source).stream(schedule.windows(), |window, raw| {
    let handle = workspace.process_chunk_in_vram(raw, window.read_len()); // stays on the device
    for event in detect(&handle)? {
        if let Some(sample) = window.remap_event(event) { /* keep */ }
    }
    Ok(())
})?;
```

### `recording`
| Type | Purpose |
|---|---|
| `RecordingSource` | `Send + Sync` trait for chunked reads. Required: `info`, `read`. Defaulted: `read_stored`, `read_native`, `chunk_samples`, `read_channel`, `read_chunk`. |
| `RecordingInfo` | Name, channels, samples, rate, stored `SampleFormat`, stored `MemoryOrder`, `start_time: RationalTime`, free-form `metadata`. |
| `ChannelInfo` | `name`, `gain`, `offset`, `unit`: `value = stored * gain + offset`, in `unit`. |
| `SignalUnit` | `Volt`, `Millivolt`, `Microvolt`, `Ampere`, `Milliampere`, `Microampere`, `Dimensionless` (default), `Other(String)`; `symbol()`. |
| `SampleFormat` | Stored little-endian type: `I8`, `I16`, `U16`, `I32`, `F32`, `F64`; `bytes`, `name`, `parse`. |
| `MemoryRecording` | In-memory channel-major source (tests, small derived signals). |
| `SlicedRecording` | Lazy view over a sample range and/or channel subset of any `Arc<dyn RecordingSource>`; no I/O on construction. |
| `check_read`, `check_read_stored` | Request validation shared by implementors. |

**Read contract:**
- `read` returns `f32`, **channel-major**, scaled to each channel's `unit`, regardless of how the
  source stores it.
- `read_stored` returns raw stored bytes (before gain/offset) so pipelines can move compact
  integers and scale on the device.
- `read_native` returns the source's own order to skip a transpose on whole-recording passes.

### `device` / `compute`
| Item | Purpose |
|---|---|
| `ComputeTarget` | `Wgpu`, `Cpu`, `Cuda`, `Hip`. One name used by CLI, Python, app and `DSP_KITCHEN_RUNTIME`. `parse`, `available`, `from_env`, `checked`. |
| `ComputeTask` + `ComputeTarget::run` | Run generic `R: Runtime` work on the selected target's default device. |
| `LaunchGeometry` | Cube dims/counts from runtime properties: `elementwise`, `channels_samples`, `per_sample`, `per_channel`, `per_row` (one cube per row, power-of-two cube for tree reductions; `row_position`), `plane_lanes`. |
| `tune` | `tune_id` (per-device autotune key), `size_class` (power-of-two bucketing). |
| `bench` | `sync`, `time_device` (median wall time including device completion). |

### `error`
`DspError`: `InvalidChannel`, `ShapeMismatch`, `InvalidSampleRate`, `BufferOverrun`,
`BufferUnderrun`, `ComputeError`, `InvalidConfig`, `SampleRange`, `Io`, `UnsupportedFormat`.

## Design rules

1. **Exact time.** Sample ↔ time conversion goes through `RationalTime` / `SampleRate` ratios,
   never accumulated `f64`.
2. **Units are data, not assumptions.** Values carry their `SignalUnit`; nothing in core assumes µV.
3. **Algorithms never pick a device.** They take a `ComputeClient<R>`; only entry points (CLI,
   Python, app) choose a `ComputeTarget`.
4. **Launch geometry comes from the runtime**, not constants.

## Consumers

`dsp-base`, `dsp-io`, `dsp-synapse`, `dsp-synapse-ml`, `dsp-view`, `dsp-stream`, `dsp-cli`,
`dsp-app`, `dsp_kitchen_py`.

## Limitations

- `TimeRange` is defined but not used by any crate yet.
