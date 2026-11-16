//! The wire protocol (no networking): message types generated from `proto/dspk_stream.proto`,
//! length-prefixed framing, and conversions to and from dspk-core / dspk-view types.
//!
//! See the `.proto` file for the session layout (control, signal and view streams).

#[cfg(not(target_family = "wasm"))]
pub mod codec;
pub mod convert;

/// Messages generated from `proto/dspk_stream.proto` (package `dspk_stream.v1`).
pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/dspk_stream.v1.rs"));
}

/// Protocol version sent in `ClientHello` and `ServerInfo`; peers of another version are refused.
pub const PROTOCOL_VERSION: u32 = 2;

/// TLS application protocol (ALPN): names the protocol and its version, so QUIC refuses peers
/// speaking another one before any message is exchanged.
pub const ALPN: &[u8] = b"dspk-stream/2";
/// The application protocol of a pairing connection (`transport::pairing`): one stream, JSON
/// messages with the same length prefix, open to computers without a certificate.
pub const PAIR_ALPN: &[u8] = b"dspk-pair/1";

/// Largest view answered: values (channels × columns, or × samples) in one envelope, so a request
/// cannot make the server allocate without bound (32 MiB of f32, under a client's message limit).
pub const MAX_VIEW_VALUES: u64 = 8 << 20;
/// Most columns a view may ask for (a screen is a few thousand pixels wide).
pub const MAX_VIEW_COLUMNS: u32 = 1 << 16;

/// Bytes of the big-endian length before every message.
pub const LENGTH_PREFIX_BYTES: usize = 4;

/// Default upper bound on one encoded message (64 MiB); larger lengths are rejected before
/// allocating.
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 64 << 20;

#[cfg(not(target_family = "wasm"))]
pub use codec::{read_message, write_message};
pub use convert::{decode_signal, envelope_from_frame, frame_from_envelope, header_from_info, info_from_header, request_from_view, view_from_request};
