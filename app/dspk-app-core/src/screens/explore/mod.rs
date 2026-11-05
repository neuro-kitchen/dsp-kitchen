//! Explore: traces, heatmaps, channels and the timeline of the open recording.

pub mod components;
pub mod explore_screen;
pub mod explore_vm;
pub mod trace;

pub use explore_screen::ExploreView;
pub use explore_vm::{ExploreEvent, ExploreVm};
pub use trace::TraceVm;
