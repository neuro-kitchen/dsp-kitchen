//! Shared data used by every module: the loaded recording and its detected events.

pub mod dataset;
pub mod events;
pub mod source;

pub use dataset::Dataset;
pub use events::SpikeEventStore;
pub use source::SignalSource;
