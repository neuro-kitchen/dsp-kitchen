//! Network transport of continuous multi-channel signals: QUIC streams with TLS carrying protobuf
//! frames, for full-fidelity processing or decimated visualization.
//!
//! Local recording I/O (formats, out-of-core reading) lives in `dsp-io`.

pub mod purpose;
pub mod network;

pub use purpose::StreamPurpose;
pub use network::{
    generate_self_signed_tls, generate_server_config, make_client_config_with_cert,
    make_insecure_client_config, QuicStreamClient, QuicStreamServer, StreamFrame,
};
