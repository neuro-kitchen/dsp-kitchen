//! Foundational primitives, timestamps, buffers, masks, and sensor layouts for `dsp-kitchen`.

pub mod error;
pub mod time;
pub mod layout;
pub mod mask;
pub mod buffer;
#[cfg(feature = "compute")]
pub mod compute;
pub mod device;
pub mod recording;
pub mod window;

// Backward-compatibility module
pub mod probe;

pub use device::{ComputeError, ComputeTarget};
pub use error::{DspError, DspResult};
pub use layout::{SensorLayout, SensorSite, Position3D, ProbeLayout};
pub use mask::ChannelMask;
pub use time::{RationalTime, SampleRate, TimeRange};
pub use buffer::{BufferChunk, MemoryLayout, MemoryOrder, SignalChunk};
pub use recording::{
    ChannelInfo, MemoryRecording, RecordingInfo, RecordingSource, SampleFormat, SlicedRecording,
};
pub use window::{ChunkSchedule, HaloWindow};
