# dsp-stream

## Intent

Network sessions for continuous multi-channel signals. A server (for example dsp-kitchen
running headless next to an acquisition system) streams a recording; a client receives its exact
description, its samples as stored, and envelopes of any window it wants to draw. Processing and
viewing run wherever the client is.

### Owns
- The session protocol (`proto/dsp_stream.proto`) and its framing.
- QUIC + TLS 1.3 endpoints, the server session and the client session.

### Must not contain
- File formats or recording I/O (dsp-io), envelopes and pyramids (dsp-view), algorithms.

## Protocol

One QUIC connection per client (ALPN `dsp-stream/1`). Every message is a 4-byte big-endian
length then its protobuf encoding; lengths above the receiver's limit are rejected before
allocating.

| Stream | Direction | Messages |
|---|---|---|
| control | client opens, both ways | `ClientHello` → `Header`; then `Subscribe { first_sample }` (again to restart) |
| signal | server → client | `SignalFrame { sequence, first_sample, samples, data, sent_unix_ns }` |
| view | one per request, both ways | `ViewRequest { channels, start, end, width }` → `EnvelopeFrame` (columns, samples or error) |

- **`Header`** is the whole `RecordingInfo`: channel names, gains, offsets and units, the sample
  rate and start time as exact fractions, the stored format, the length and the metadata.
- **`SignalFrame.data`** holds the stored words, channel-major, as the header's format. Integer
  recordings travel at their stored size and are scaled by the client (`decode_signal`, using
  dsp-core's `SampleFormat::decode`).
- **Views** travel on their own streams, so a viewer's zoom never waits behind signal data. The
  server answers with dsp-view's `View::read`: from its pyramid when it has one.

The types are generated at build time by `prost-build` with `protox` (no `protoc` needed).

## Usage

```rust,ignore
use dsp_stream::{client_config, serve_recording, server_config, ServeOptions, Server, ServerIdentity, ServerTrust, Session};

// Server
let tls = server_config(&ServerIdentity::Files { certificate_chain: cert.into(), private_key: key.into() })?;
let server = Server::bind("0.0.0.0:50051".parse()?, tls.config)?;
while let Some(connection) = server.accept().await {
    tokio::spawn(serve_recording(connection, source.clone(), Some(pyramid.clone()), ServeOptions::default()));
}

// Client
let config = client_config(&ServerTrust::from_file(ca_path)?)?;
let mut session = Session::connect(addr, "acquisition.lab", config).await?;
let info = session.info();                       // channels, units, exact rate
let envelope = session.view(&view).await?;       // [min, max] columns or samples
let mut signal = session.subscribe(0).await?;
while let Some((first_sample, values)) = signal.next_values().await? { /* channels × n */ }
```

## Reference

| Item | What |
|---|---|
| `ServerIdentity::SelfSigned { names }` / `Files { certificate_chain, private_key }` | The server's certificate: generated (development) or PEM files. |
| `ServerTrust::Certificates` / `from_file` / `DangerAcceptAnyCertificate` | What a client accepts; the last only on a fully trusted network. |
| `server_config`, `client_config` | QUIC configurations (TLS 1.3, ALPN set). |
| `Server::{bind, accept, local_addr, close}` | Listening endpoint. |
| `serve_recording(connection, source, pyramid, ServeOptions)` | Serves one client until it leaves. |
| `ServeOptions` | `frame_sec` (`DEFAULT_FRAME_SEC` = 0.02 s), `pacing` (`RealTime` / `Unpaced`), `loop_playback`, `queue_frames` (`DEFAULT_QUEUE_FRAMES` = 8, bounds memory), `max_message_bytes`. |
| `Session::{connect, info, subscribe, view, close}` | Client session. |
| `SignalReceiver::{next_frame, next_values}` | Frames as sent, or scaled channel-major values. |
| `StreamError` | I/O, TLS, connect, connection, read / write, too large, truncated, decode, protocol, DSP. |

## Limitations

- No client authentication yet (mutual TLS): anyone who reaches the port and trusts the server
  can receive the recording. Bind to a private network.
- Servers replay recordings (`RecordingSource`); live acquisition producers are not written yet.
- A client cannot yet subscribe to a subset of channels.
