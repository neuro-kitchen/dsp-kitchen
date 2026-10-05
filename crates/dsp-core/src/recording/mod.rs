//! Format-independent continuous recordings: descriptor, chunked read trait, in-memory source, and lazy slicing.
//!
//! File formats implement [`RecordingSource`] in `dsp-io`; processing and UI code read through
//! the trait and never depend on a concrete format.

pub mod format;
pub mod info;
pub mod memory;
pub mod slice;
pub mod source;
pub mod unit;

pub use format::SampleFormat;
pub use info::{ChannelInfo, RecordingInfo};
pub use memory::MemoryRecording;
pub use slice::SlicedRecording;
pub use source::{check_read, check_read_stored, RecordingSource};
pub use unit::SignalUnit;
