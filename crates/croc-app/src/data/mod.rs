//! Shared data used by every module: the open recording and its detected events.

pub mod dataset;
pub mod events;

pub use dataset::Dataset;
pub use events::SpikeEventStore;
