//! Storage containers shared by every format: how bytes are stored, independent of what they mean.

pub mod binary;
pub mod npy;
#[cfg(feature = "zarr")]
pub mod zarr;
