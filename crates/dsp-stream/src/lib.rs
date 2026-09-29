//! High-throughput transport, streaming circular ring buffers, Zarr v3, and visualization decimation.

pub mod decimate;
pub mod purpose;
pub mod zarr;
pub mod ring;

pub use purpose::StreamPurpose;
pub use decimate::min_max_decimate;
pub use zarr::{create_zarr_recording, read_zarr_recording};
pub use ring::MultiChannelRingBuffer;
