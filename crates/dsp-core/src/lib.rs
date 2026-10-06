//! Foundational primitives, timestamps, buffers, and masks for `dsp-kitchen`.

pub mod error;
pub mod time;
pub mod mask;
pub mod buffer;
#[cfg(feature = "compute")]
pub mod compute;
pub mod device;
pub mod recording;
pub mod window;

pub use device::{ComputeError, ComputeTarget};
pub use error::{DspError, DspResult};
pub use mask::ChannelMask;
pub use time::{RationalTime, SampleRate, TimeRange};
pub use buffer::{BufferLayout, MemoryOrder, SignalChunk};
pub use recording::{
    ChannelInfo, MemoryRecording, RecordingInfo, RecordingSource, SampleFormat, SignalUnit, MICROVOLTS_PER_VOLT,
    SlicedRecording,
};
pub use window::{ChunkSchedule, HaloWindow, WindowLoader};
