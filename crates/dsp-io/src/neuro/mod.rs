//! Neural-recording formats, probe geometry, and synthetic recordings.

pub mod mtscomp;
#[cfg(feature = "zarr")]
pub mod nwb;
pub mod probe;
pub mod spikeglx;
pub mod synthetic;
