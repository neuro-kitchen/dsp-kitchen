//! NWB files stored as Zarr v3 (hdmf-zarr layout).

pub mod acquisition;
mod format;
pub mod units;

pub use acquisition::{is_nwb_zarr, list_series, NwbZarrRecording, SeriesEntry};
pub use format::Nwb;
pub use units::infer_nwb_sample_rate;
