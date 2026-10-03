//! Views: thin renderers of the view models, forwarding input to them.

pub mod explore;
pub mod panels;
// The plot kit serves the Curation views (step 8)
#[allow(dead_code)]
pub mod plot;
pub mod trace;

pub use explore::ExploreView;
