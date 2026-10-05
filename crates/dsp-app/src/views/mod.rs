//! Views: thin renderers of the view models, forwarding input to them.

pub mod curation;
pub mod explore;
pub mod panels;
// Parts of the plot kit (axes, lasso) serve Curation views still to come
#[allow(dead_code)]
pub mod plot;
pub mod trace;

pub use curation::CurationView;
pub use explore::ExploreView;
