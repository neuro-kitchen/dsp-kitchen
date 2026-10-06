//! Out-of-core sliding window scheduling, data loading, prefetching, and buffer slicing.
//!
//! `dsp-orchestrate` bridges [`dsp_core`] schedule descriptors and [`dsp_io`] recording sources
//! into a unified, high-performance runtime for out-of-core signal processing algorithms.

pub mod buffer;
pub mod coord;
pub mod loader;

pub use buffer::WindowBuffer;
pub use coord::remap_event;
pub use loader::WindowLoader;

// Re-export foundational window descriptors from dsp-core for convenience
pub use dsp_core::{ChunkSchedule, HaloWindow};
