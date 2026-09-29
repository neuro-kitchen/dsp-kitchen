//! High-throughput transport, Shared Memory v2 IPC, gRPC/QUIC, and visualization decimation.

pub mod decimate;
pub mod purpose;
pub mod zarr;

pub use purpose::StreamPurpose;
pub use decimate::min_max_decimate;
pub use zarr::{create_zarr_recording, read_zarr_recording};

