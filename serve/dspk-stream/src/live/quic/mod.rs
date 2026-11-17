//! The live protocol over QUIC + TLS 1.3 (ALPN `dspk-live/1`), framed as the session protocol
//! (a 4-byte big-endian length, then the protobuf encoding).
//!
//! - Device side: accept connections with [`crate::transport::Server`] (pairing, client
//!   certificates: [`crate::transport::server_config_live`]) and hand each to [`serve_device`].
//! - Client side: [`DeviceClient`].

mod client;
mod server;

pub use client::{DeviceClient, TapReceiver, ViewReceiver};
pub use server::serve_device;

/// Whether `connection` speaks the live protocol (a device serves the session protocol too, for
/// its finished recordings: [`crate::transport::serve_session`]).
pub fn is_live(connection: &quinn::Connection) -> bool {
    connection
        .handshake_data()
        .and_then(|d| d.downcast::<quinn::crypto::rustls::HandshakeData>().ok())
        .and_then(|d| d.protocol)
        .is_some_and(|p| p == super::ALPN)
}
