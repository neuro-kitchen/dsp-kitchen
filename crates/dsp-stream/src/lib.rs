//! Network sessions for continuous multi-channel signals: a server streams a recording, a client
//! receives its exact description, its stored samples, and envelopes of any view.
//!
//! - [`protocol`]: the messages (generated from `proto/dsp_stream.proto`), their framing, and
//!   conversions to and from dsp-core / dsp-view types.
//! - [`transport`]: QUIC + TLS 1.3 endpoints, the server session and the client session.
//!
//! Local recording I/O lives in `dsp-io`; envelopes and pyramids in `dsp-view` (a server answers
//! views with them).

pub mod error;
pub mod protocol;
pub mod transport;

pub use error::{StreamError, StreamResult};
pub use transport::{
    client_config, serve_recording, server_config, Pacing, ServeOptions, Server, ServerIdentity, ServerTls, ServerTrust, Session, SignalReceiver,
};
