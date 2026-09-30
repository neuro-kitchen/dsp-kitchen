//! High-throughput continuous streaming and lock-free ring buffering (data transport).
//!
//! File formats live in `dsp-io`; the min/max decimation kernel lives in `dsp-base`.

pub mod buffer;
pub mod purpose;
pub mod network;

// Convenient re-exports
pub use buffer::MultiChannelRingBuffer;
pub use purpose::StreamPurpose;
pub use network::{
    generate_self_signed_tls, generate_server_config, make_client_config_with_cert,
    make_insecure_client_config, QuicStreamClient, QuicStreamServer, StreamFrame,
};


// Backward-compatible module aliases
pub use buffer as ring;
