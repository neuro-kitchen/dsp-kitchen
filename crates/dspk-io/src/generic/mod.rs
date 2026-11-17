//! Domain-independent recording formats.

pub mod raw;
pub mod wav;
pub mod zarr_options;
#[cfg(feature = "zarr-read")]
pub mod zarr_traces;
