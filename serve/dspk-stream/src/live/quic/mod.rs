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
