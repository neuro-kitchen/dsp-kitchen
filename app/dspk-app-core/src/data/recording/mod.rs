//! Shared data used by every module: the open recording and its spike events.

pub mod dataset;
pub mod events;
/// A view's processed signal and the chains kept for a recording (dspk-pipeline's, shared with
/// `dspk serve` and a page's engine).
pub mod processed {
    pub use dspk_pipeline::view::{Made, PreviewSource, ProcessedChains, ProcessedSignal};
}
pub mod sources;

pub use dataset::Dataset;
pub use events::SpikeEventStore;
pub use sources::SourceSet;
