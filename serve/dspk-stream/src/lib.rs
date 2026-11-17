//! Network sessions for continuous multi-channel signals: a server streams a recording, a client
//! receives its exact description, its stored samples, and envelopes of any view.
//!
//! - [`protocol`]: the messages (generated from `proto/dspk_stream.proto`), their framing, and
//!   conversions to and from dspk-core / dspk-view types.
//! - [`transport`]: QUIC + TLS 1.3 endpoints, the server session and the client session.
//! - [`live`]: devices (live sources, taps, a director): draft, messages only.
//!
//! Local recording I/O lives in `dspk-io`; envelopes and pyramids in `dspk-view` (a server answers
//! views with them).

pub mod engine;
pub mod error;
/// Devices (live sources, taps, a director): not in a browser page yet.
#[cfg(not(target_family = "wasm"))]
pub mod live;
pub mod pair;
pub mod plugin;
pub mod protocol;
pub mod remote;
/// A browser page's connection to its server (a WebSocket), as a [`Link`].
#[cfg(target_family = "wasm")]
pub mod web_client;
/// A page's connection to its engine worker (`dspk-engine-web`), as a [`Link`].
#[cfg(target_family = "wasm")]
pub mod worker_client;
#[cfg(not(target_family = "wasm"))]
pub mod transport;

pub use error::{StreamError, StreamResult};
pub use plugin::{PluginContext, ServerPlugin};
pub use remote::{plugin_call, Link, RemoteSignal, RemoteSource};
/// Files under a server's data root (safe paths, listings).
#[cfg(not(target_family = "wasm"))]
pub use transport::files;
/// Jobs a server runs for its clients (sorts), one at a time.
#[cfg(not(target_family = "wasm"))]
pub use transport::jobs;
#[cfg(not(target_family = "wasm"))]
pub use transport::{
    checked_view, client_config, client_config_with, serve_recording, serve_session, server_config, Device, OpenRecording, RecordingOpener, ViewProcessor, server_config_with_clients, Authority, ClientIdentity, ClientRecord,
    Enrollment, Pacing, Peer, Permission, ServeOptions, Server, ServerIdentity, ServerTls, ServerTrust, Session, SignalReceiver, DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_REQUEST_BYTES, MAX_VIEW_COLUMNS, MAX_VIEW_VALUES, REFUSED_CLIENT,
};
