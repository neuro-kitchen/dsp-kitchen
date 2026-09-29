//! High-throughput continuous streaming, lock-free ring buffering, chunked Zarr v3 persistence, and LOD decimation.

pub mod buffer;
pub mod storage;
pub mod reduction;
pub mod purpose;

// Convenient re-exports
pub use buffer::MultiChannelRingBuffer;
pub use storage::{create_zarr_recording, read_zarr_recording};
pub use reduction::min_max_decimate;
pub use purpose::StreamPurpose;

// Backward-compatible module aliases
pub use reduction as decimate;
pub use storage as zarr;
pub use buffer as ring;
