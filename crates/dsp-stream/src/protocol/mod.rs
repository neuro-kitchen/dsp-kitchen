//! The wire protocol (no networking): message types generated from `proto/dsp_stream.proto`,
//! length-prefixed framing, and conversions to and from dsp-core / dsp-view types.
//!
//! See the `.proto` file for the session layout (control, signal and view streams).

pub mod codec;
pub mod convert;

/// Messages generated from `proto/dsp_stream.proto` (package `dsp_stream.v1`).
pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/dsp_stream.v1.rs"));
}

/// Protocol version sent in `ClientHello` and `Header`; peers of another version are refused.
pub const PROTOCOL_VERSION: u32 = 1;

/// TLS application protocol (ALPN): names the protocol and its version, so QUIC refuses peers
/// speaking another one before any message is exchanged.
pub const ALPN: &[u8] = b"dsp-stream/1";

/// Bytes of the big-endian length before every message.
pub const LENGTH_PREFIX_BYTES: usize = 4;

/// Default upper bound on one encoded message (64 MiB); larger lengths are rejected before
/// allocating.
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 64 << 20;

pub use codec::{read_message, write_message};
pub use convert::{decode_signal, envelope_from_frame, frame_from_envelope, header_from_info, info_from_header, request_from_view, view_from_request};
