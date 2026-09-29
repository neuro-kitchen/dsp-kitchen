//! Model layer for croc-app.
//!
//! Encapsulates domain models and pure business logic:
//! - `Dataset`: Binary/synthetic recording buffer and metadata
//! - `TimelineState`: Interactive playback, scrubbing, and temporal coordinates
//! - `decimation`: Constant O(W) screen-space Min-Max LOD decimation
//! - `events`: Action potential detection and timeline event markers

pub mod dataset;
pub mod decimation;
pub mod events;
pub mod timeline;

pub use dataset::Dataset;
#[allow(unused_imports)]
pub use decimation::{decimate_min_max, LodBucket};
pub use events::SpikeEventStore;
pub use timeline::TimelineState;
