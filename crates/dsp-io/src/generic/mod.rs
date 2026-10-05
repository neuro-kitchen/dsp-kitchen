//! Domain-independent recording formats.

pub mod raw;
#[cfg(feature = "zarr")]
pub mod zarr_traces;
