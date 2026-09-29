//! Foundational primitives, timestamps, buffers, masks, and sensor layouts for `dsp-kitchen`.

pub mod error;
pub mod time;
pub mod layout;
pub mod mask;
pub mod buffer;
pub mod device;

// Backward-compatibility module
pub mod probe;

pub use error::{DspError, DspResult};
pub use layout::{SensorLayout, SensorSite, Position3D, ProbeLayout};
pub use mask::ChannelMask;
pub use time::{RationalTime, SampleRate, TimeRange};
pub use buffer::{BufferChunk, MemoryLayout};
