//! Neural-recording formats, spike-sorting files (Phy / Kilosort, NWB units, `.sorting.zarr`),
//! probe geometry, and synthetic recordings.

pub mod mtscomp;
#[cfg(feature = "zarr")]
pub mod nwb;
pub mod phy;
pub mod probe;
mod sorting_format;
#[cfg(feature = "zarr")]
pub mod sorting_zarr;
pub mod spikeglx;
pub mod synthetic;
pub mod templates;

pub use sorting_format::{detect_sorting, SortingFormat};
