# dsp-stream review notes (working file)

Reviewed 2026-10-05 on `versions/v0.14` (prompted by dsp-synapse depending on it). No changes yet.

Intended role (user): network communication — e.g. the app installed headless on a server streams
data; algorithms run locally.

## Inventory (~1.15k lines)
| Module | What | Users |
|---|---|---|
| `reader/prefetch.rs` (`PrefetchReader`) | Double-buffered **local file** reading of `RecordingSource` halo windows on a background thread (disk / decompression overlapped with GPU work). No network. | dsp-synapse (`streaming/runner.rs`), dsp-cli (`serve.rs`) |
| `buffer/ring.rs` (`MultiChannelRingBuffer`, `OverrunPolicy`) | Channel-major circular buffer for live ingestion. | **none** |
| `network/` (QUIC transport, TLS, protobuf `StreamFrame`) | Over-the-network streaming. | dsp-cli (`serve`, `receive`, `stream`) |
| `purpose.rs` (`StreamPurpose`) | Processing vs visualization (decimated LOD) stream. | dsp-cli |

dsp-base does **not** depend on dsp-stream. dsp_kitchen_py lists it in `Cargo.toml` but uses nothing.

## Findings
- SR1 **`PrefetchReader` is I/O, not networking**: it belongs in dsp-io (next to `RecordingSource`,
  `CachedRecording`, `open`). It is the only reason dsp-synapse depends on dsp-stream.
- SR2 `MultiChannelRingBuffer` has no users (`receive.rs` does not use it).
- SR3 Unused dependencies: `shared_memory`, `memmap2`, `tonic`, `serde_json`; dsp_kitchen_py's
  dependency on dsp-stream is unused.
- SR4 `StreamFrame` bakes in µV (`gain_uv`, int16 payload "value in µV"); should carry the stored
  `SampleFormat` and `SignalUnit` like core's `ChannelInfo`. Only f32 / int16 payloads.
- SR5 `lib.rs`: backward-compatibility alias `pub use buffer as ring`; doc says "the min/max decimation
  kernel lives in dsp-base" (now parked for dsp-app — stale).
- SR6 `network::tls::make_insecure_client_config` (skips server certificate verification) is public
  API with no "development only" guard.
- SR7 Scope: the network layer moves raw frames only (serve / receive); there is no remote job
  execution, so "run algorithms locally on a headless server" is not implemented here.

## Changes (2026-10-05)

| Change | Downstream action |
|---|---|
| `reader/prefetch.rs` (`PrefetchReader`) **moved** to `crates/dsp-io/src/core/prefetch.rs`, exported as `dsp_io::PrefetchReader` (code unchanged; doc no longer says µV). `reader/` removed. | dsp-cli `commands/serve.rs`: `dsp_stream::PrefetchReader` → `dsp_io::PrefetchReader`. dsp-synapse updated (`streaming/runner.rs`). |
| `buffer/` (`MultiChannelRingBuffer`, `OverrunPolicy`) **parked** in `refactoring/dsp-stream/buffer/` (no users). | None. dsp-core's `DspError::BufferOverrun` / `BufferUnderrun` lose their only user (revisit). |
| `lib.rs`: alias `pub use buffer as ring` and the stale min/max-kernel doc removed; module doc now states the network role. | — |
| `Cargo.toml`: removed unused `shared_memory`, `memmap2`, `tonic`, `serde_json`, and `dsp-core` (only the ring buffer used it). | — |
| dsp-synapse `Cargo.toml`: `dsp-stream` → `dsp-io` (features `zarr`, `neuro`). | — |
| dsp_kitchen_py `Cargo.toml`: unused `dsp-stream` dependency removed. | — |

Deferred to the full dsp-stream review: SR4 (µV in `StreamFrame`), SR6 (insecure TLS client guard),
SR7 (remote execution scope).
