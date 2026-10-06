//! Orchestration of recordings larger than memory: halo window scheduling and streaming.
//!
//! - [`ChunkSchedule`] splits a sample range into non-overlapping windows, each padded with halos
//!   (filter settling, lookback, lookahead) and clamped to the recording ([`HaloWindow`]).
//! - [`WindowLoader`] streams any list of windows of a [`crate::RecordingSource`], reading the next
//!   window on a background thread while the current one is processed.
//! - [`HaloWindow::remap_event`] maps an event found in the padded buffer back to a recording
//!   sample, keeping it only in the window that owns it.

mod loader;
mod schedule;

pub use loader::WindowLoader;
pub use schedule::{ChunkSchedule, HaloWindow};
