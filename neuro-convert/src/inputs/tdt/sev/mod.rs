//! SEV files: per-channel stream files written by the RS4 Data Streamer or by Synapse's
//! "Discrete Files" option (OpenEx: `Stream_Store_MC2`). Preferred over TEV packets when both
//! exist, as in TDT's own reader.

pub mod files;
pub mod header;
pub mod log;
pub mod recording;

pub use files::{group, SevStores};
pub use recording::SevStream;
