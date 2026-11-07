//! Pipeline: the session's processing graph on an infinite canvas (sources, ops, views,
//! writes), the operations tree and the inspector.

pub mod canvas;
pub mod catalog;
pub mod channel_table;
pub mod pipeline_screen;

pub use pipeline_screen::PipelineView;
