//! Classical deterministic DSP and filtering algorithms powered by CubeCL kernels.

pub mod geometry;
pub mod filter;
pub mod spatial;
pub mod math;
pub mod pipeline;

pub use geometry::LaunchGeometry;
pub use pipeline::{Pipeline, PipelineStage};
