//! Shared data used by every module: the open recording and its spike events.

pub mod dataset;
pub mod events;
/// A view's processed signal (dspk-pipeline's, shared with `dspk serve`).
pub mod processed {
    pub use dspk_pipeline::processed::*;
}
pub mod sources;

pub use dataset::Dataset;
pub use events::SpikeEventStore;
pub use sources::SourceSet;
