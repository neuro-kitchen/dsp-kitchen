//! Format-independent continuous recordings: descriptor, chunked read trait, in-memory source.
//!
//! File formats implement [`RecordingSource`] in `dsp-io`; processing and UI code read through
//! the trait and never depend on a concrete format.

pub mod format;
pub mod info;
pub mod memory;
pub mod source;

pub use format::SampleFormat;
pub use info::{ChannelInfo, RecordingInfo};
pub use memory::MemoryRecording;
pub use source::{check_read, RecordingSource};
