//! Out-of-core chunked streaming readers over [`dsp_core::RecordingSource`].

pub mod prefetch;

pub use prefetch::PrefetchReader;
