//! Model layer for croc-app.
//!
//! Encapsulates domain models and pure business logic:
//! - `SignalSource`: Read-only data source trait consumed by views
//! - `Dataset`: Memory-mapped / synthetic recording implementing `SignalSource`
//! - `TimelineState`: Interactive playback, scrubbing, and temporal coordinates
//! - `events`: Action potential detection (via `dsp-synapse`) and timeline event markers

pub mod dataset;
pub mod events;
pub mod source;
pub mod timeline;

pub use dataset::Dataset;
pub use events::SpikeEventStore;
pub use source::SignalSource;
pub use timeline::TimelineState;
