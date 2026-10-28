//! Shared data used by every module: the open recording and its spike events.

pub mod dataset;
pub mod events;
pub mod sources;

pub use dataset::Dataset;
pub use events::SpikeEventStore;
pub use sources::SourceSet;
