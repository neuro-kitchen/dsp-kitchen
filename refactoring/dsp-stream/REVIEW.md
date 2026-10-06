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

## Second review (2026-10-06): what is left (~760 lines)

Read-only; no code changed. What remains is `purpose.rs` plus `network/{frame, tls, transport}.rs`.
SR4, SR6 and SR7 above are still open. **D** = a decision needed from you.

### Protocol

- **ST1 — No stream header; every frame repeats the rate.** A client learns channels and rate
  from the first frame. It never gets channel names, gains, offsets, units, the stored format, the
  probe, or the start time. A received stream cannot become a `RecordingInfo`, so it cannot feed a
  pipeline or be saved as a recording dsp-io can open.
  *Proposal:* the first message on the stream is a `StreamHeader` carrying the `RecordingInfo`
  fields (with the `SampleRate` as an exact rational, not `f64`) and the protocol version; frames
  then carry only sequence, first sample and payload. **D**
- **ST2 — The payload is re-quantized, not stored (SR4).** `serve --int16-gain-uv` reads scaled
  f32 and rounds it to int16 with a user-given gain, which is lossy and assumes µV.
  `RecordingSource::read_stored` already gives the stored words.
  *Proposal:* frames carry the stored `SampleFormat` words. Gain, offset and unit go in the header,
  and the client scales, on the device for pipelines, as dsp-base does for local files.
- **ST3 — No schema file.** `StreamFrame` is a hand-written `prost::Message`; there is no `.proto`,
  so a Python or app client must copy the field tags by hand. *Proposal:* a `.proto` file in the
  crate with `prost-build` generating the types, versioned. **D**
- **ST4 — `StreamPurpose::Visualization` is not real.**
  - Nothing produces it: the min/max decimation was parked for dsp-app, and `serve` always sends
    `Processing`.
  - `target_points_per_channel` is not sent; it is rebuilt from `samples`.
  - The CLI's `stream` command prints a "Min-Max Envelope" kernel that does not exist.

  *Decided 2026-10-06:* decimation lives in the new crate `dsp-view`; dsp-stream carries its
  view requests and envelopes (DEC3).
- **ST5 — A truncated frame is treated as end of stream.** `recv_frame_limited` returns `Ok(None)`
  when the stream ends **after** a length prefix (mid-frame). Only an end before the prefix is a
  clean end; a mid-frame end should be an error.

### Transport and TLS

- **ST6 — Bind and connect errors are discarded.** `Endpoint::server` and `connect` errors are
  mapped to `ConnectionError::LocallyClosed`, so "address in use" or an invalid name reads as
  "locally closed". Errors elsewhere are `Box<dyn Error>`. *Proposal:* a `StreamError` enum (io,
  tls, connect, connection, frame, decode).
- **ST7 — Duplicated TLS code.** `generate_self_signed_tls`, `generate_server_config` and
  `make_client_config_with_cert` repeat the same builders, and the ALPN literal `dsp-stream-quic`
  appears four times. *Proposal:* one server builder and one client builder, plus an `ALPN`
  constant (that also versions the protocol).
- **ST8 — Certificates and access.**
  - The server can only use a new self-signed certificate per run, so clients must fetch a new
    `.der` after every restart. There is no option to load a certificate and key from files.
  - There is no client authentication: anyone who reaches the UDP port receives the recording.
  - For the intended use (a headless server streaming neural data), add a certificate / key
    from files, and optionally client certificates (mutual TLS), and keep the insecure client
    behind an explicitly named option (SR6). **D**
- **ST9 — Overstated docs.** The module doc claims "zero head-of-line blocking" and "connection
  migration", but one ordered stream carries every frame (so head-of-line blocking applies).
  *Proposal:* describe what it is: one reliable, ordered QUIC stream per client.
- **ST10 — The client binds `0.0.0.0:0`**, so it cannot reach an IPv6 server. *Proposal:* bind the
  unspecified address of the server's family.

### Scope (SR7, restated)

You described the role as: the app installed headless on a server streams data; algorithms
run locally. Missing for that:
1. a session (header, then frames), as in ST1;
2. a client-side **live source** that turns received frames into chunks a `PipelineWorkspace`
   can process (stored words uploaded and scaled on the device);
3. live-acquisition producers, not only file replay, which is all `serve` does today.

*Proposal:* dsp-stream owns 1 and 2 (it stays free of algorithms); producers are adapters in
dsp-io or the app. **D**

### Fine as is

- Length prefixes are checked against `DEFAULT_MAX_FRAME_BYTES` before allocating.
- `validate()` is called on every decoded frame and rejects two payloads or a size mismatch.
- The tests cover the end-to-end transfer, the insecure client and an oversized prefix.

## Changes (2026-10-06): tree and decimation

| Change | Downstream action |
|---|---|
| `network/frame.rs` → `protocol/frame.rs`; `purpose.rs` → `protocol/purpose.rs` (`protocol` = wire format, no I/O). | `dsp_stream::network::StreamFrame` / `dsp_stream::purpose::StreamPurpose` → `dsp_stream::protocol::…` (dsp-cli updated). |
| `network/tls.rs` → `transport/tls.rs`; `network/transport.rs` split into `transport/server.rs` (`QuicStreamServer`) and `transport/client.rs` (`QuicStreamClient`), with the tests in `transport/mod.rs`. `network/` removed. | `dsp_stream::network::…` → `dsp_stream::transport::…` (dsp-cli updated). |
| **Decimation un-parked into dsp-stream** (user, 2026-10-06: the app may run locally or online, so envelopes are built next to the data or across the network). `refactoring/dsp-base/resampler/{mod, minmax, decimate, summary, cache, summarize}.rs` → `src/decimation/`. Code unchanged, except `peak_to_peak` removed (duplicate of `dsp_base::math::peak_to_peak`; its test now covers `mean_range` only) and the module doc. | dsp-app: `dsp_base::resampler::…` → `dsp_stream::decimation::…` (`min_max_decimate(_into)`, `MinMaxCache`, `CacheIdentity`, `cache_path`, `MinMaxSummary`, `Summarizer`, `Progress`, `OnProgress`, `Block`, `Columns`). |
| `Cargo.toml`: `dsp-core`, `rayon`, `memmap2` added (used by decimation). | — |
| **Same day: decimation moved out again** into the new crate `dsp-view` (user: keep viewing code in one crate, CPU and GPU, not split across crates). dsp-stream's dependencies are back to transport only. | dsp-app: `dsp_view::…` (see `refactoring/dsp-view/README.md`). |

`cargo check -p dsp-stream --all-targets` passes; no warnings. Tests not run (end pass).

Still open for decimation (now tracked in `refactoring/dsp-view/README.md`):
- **DEC1:** two pyramid bases (`summary::BASE = 256`, `cache::DEFAULT_BASE = 64`); unify them.
- **DEC2:** host only (rayon). A device envelope kernel is needed for data already on the device
  (a local app after a `PipelineWorkspace`).
- **DEC3:** an envelope frame for remote viewers: `StreamPurpose::Visualization` + `[min, max]`
  columns (ST1, ST4).

## Changes (2026-10-06): session protocol

| Finding | Change |
|---|---|
| ST1, ST3 | `proto/dsp_stream.proto` (package `dsp_stream.v1`) defines the session, and `build.rs` generates the types with `prost-build` + **`protox`**, so building needs no `protoc`. Streams: **control** (client → `ClientHello`, server → `Header` = the full `RecordingInfo` with exact rate / start-time fractions, channels with gain, offset and unit, stored format, metadata; then `Subscribe { first_sample }`), **signal** (one-way, `SignalFrame { sequence, first_sample, samples, data, sent_unix_ns }`), **view** (one two-way stream per `ViewRequest` → `EnvelopeFrame` = columns / samples / error). Views never wait behind signal data. |
| ST2 | Frames carry stored words (`read_stored`); sources without stored reads send `f32` with the header saying so (gain 1, offset 0). Clients scale with `decode_signal` → dsp-core `SampleFormat::decode`. No µV assumption, no re-quantization. |
| ST4 | `StreamPurpose` removed: viewers send `View` requests; servers answer with dsp-view `View::read` (pyramid when given). |
| ST5 | `protocol::codec::read_message`: `Ok(None)` only for a clean end before a message; `StreamError::Truncated` inside one; lengths over the limit are rejected before allocating. |
| ST6 | `StreamError` (io, tls, connect, connection, read, write, too large, truncated, decode, protocol, dsp); bind / connect errors are kept, not replaced by "locally closed". |
| ST7 | One `server_config(&ServerIdentity)` and one `client_config(&ServerTrust)`; the ALPN is the constant `protocol::ALPN = "dsp-stream/1"` (it also versions the protocol, with `PROTOCOL_VERSION`). |
| ST8 (part) | `ServerIdentity::{SelfSigned { names }, Files { certificate_chain, private_key }}` (PEM); `ServerTrust::{Certificates(DER…), DangerAcceptAnyCertificate}` + `ServerTrust::from_file` (PEM or DER). The accept-any verifier still checks handshake signatures. **Open:** client certificates (mutual TLS). |
| ST9 | Docs describe what it is: one connection per client, separate streams for control, signal and views. |
| ST10 | `Session::connect` binds the unspecified address of the server's family (IPv4 or IPv6). |

**API:** `Server::{bind, local_addr, accept, close}`; `serve_recording(connection, source,
pyramid, ServeOptions)` with `ServeOptions { frame_sec (DEFAULT_FRAME_SEC = 0.02), pacing
(RealTime / Unpaced), loop_playback, queue_frames (DEFAULT_QUEUE_FRAMES = 8),
max_message_bytes (DEFAULT_MAX_MESSAGE_BYTES = 64 MiB) }`. A reader thread fills a bounded queue
(memory bounded whatever the length). `Session::{connect, info, subscribe, view, close}`;
`SignalReceiver::{next_frame, next_values}`.

**Dependencies:** dsp-core, dsp-view (no default features: host envelopes only), prost, quinn,
rcgen, rustls, rustls-pki-types (`std`, PEM), thiserror, tokio, tracing. Build: prost-build,
protox. `serde` dropped.

**Tests (written, not run):** codec / conversions (header round trip exact incl. `Other`
unit and fractional rate, signal decoding, views and envelopes); end to end over QUIC (the whole
signal from a sample offset arrives in order and equals `read`; views from the pyramid equal
`View::read`, short views return samples, a bad view returns a server error; an untrusted
certificate is refused and `DangerAcceptAnyCertificate` connects).

`cargo check` / `clippy` / `doc` for dsp-stream: clean.

**Downstream:** dsp-cli `net/serve.rs` and `net/receive.rs` use the removed `StreamFrame`,
`QuicStreamServer` / `QuicStreamClient` and TLS helpers: they are to be rewritten on
`serve_recording` / `Session` (CLI3, CLI4). dsp-app: same, for any remote viewing.
