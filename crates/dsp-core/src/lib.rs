//! Foundational primitives, timestamps, buffers, and probe geometries for `dsp-kitchen`.

pub mod error;
pub mod time;
pub mod probe;
pub mod buffer;
pub mod device;

pub use error::{DspError, DspResult};
