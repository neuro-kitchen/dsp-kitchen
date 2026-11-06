//! Network sessions for continuous multi-channel signals: a server streams a recording, a client
//! receives its exact description, its stored samples, and envelopes of any view.
//!
//! - [`protocol`]: the messages (generated from `proto/dspk_stream.proto`), their framing, and
//!   conversions to and from dspk-core / dspk-view types.
//! - [`transport`]: QUIC + TLS 1.3 endpoints, the server session and the client session.
//!
//! Local recording I/O lives in `dspk-io`; envelopes and pyramids in `dspk-view` (a server answers
//! views with them).

pub mod error;
pub mod protocol;
pub mod transport;

pub use error::{StreamError, StreamResult};
/// Files under a server's data root (safe paths, listings).
pub use transport::files;
pub use transport::{
    checked_view, client_config, client_config_with, serve_recording, serve_session, server_config, Device, OpenRecording, RecordingOpener, server_config_with_clients, Authority, ClientIdentity, ClientRecord,
    Enrollment, Pacing, Peer, Permission, ServeOptions, Server, ServerIdentity, ServerTls, ServerTrust, Session, SignalReceiver, DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_REQUEST_BYTES, MAX_VIEW_COLUMNS, MAX_VIEW_VALUES, REFUSED_CLIENT,
};
