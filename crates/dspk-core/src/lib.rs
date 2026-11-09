//! Foundational primitives, timestamps, buffers, and masks for `dspk`.
//!
//! # Start here
//!
//! | Need | Look at |
//! |---|---|
//! | Choose a device | [`ComputeTarget`]: `ComputeTarget::from_env()?.client()?` picks the runtime named by `DSPK_RUNTIME`, else the first compiled in |
//! | Describe or read a recording | [`RecordingSource`], [`RecordingInfo`], [`MemoryRecording`] |
//! | Walk a recording larger than memory | [`ChunkSchedule`], [`HaloWindow`], [`WindowLoader`] |
//! | Exact sample rates and times | [`SampleRate`], [`RationalTime`], [`TimeRange`] |
//!
//! Guides: <https://neuro-kitchen.github.io/dsp-kitchen/>.

pub mod error;
pub mod time;
pub mod mask;
pub mod buffer;
#[cfg(feature = "compute")]
pub mod compute;
pub mod device;
pub mod progress;
pub mod recording;
pub mod window;

pub use device::{ComputeError, ComputeTarget};
pub use error::{DspError, DspResult};
pub use mask::ChannelMask;
pub use time::{RationalTime, SampleRate, TimeRange};
pub use buffer::{BufferLayout, MemoryOrder, SignalChunk};
pub use recording::{
    ChannelInfo, MemoryRecording, RecordingInfo, RecordingSource, SampleFormat, SignalUnit, MICROVOLTS_PER_VOLT,
    SlicedRecording, ChannelMap, ChannelMapFile, ChannelPair, RemappedSource,
};
pub use progress::{NoProgress, ProgressEvent, ProgressSink};
pub use window::{ChunkSchedule, HaloWindow, WindowLoader};
