//! Classical deterministic DSP, linear algebra, and filtering algorithms powered by CubeCL kernels.

pub mod compute;
pub mod geometry;
pub mod filter;
pub mod spatial;
pub mod math;
pub mod linalg;
pub mod pipeline;
pub mod reduction;

pub use compute::{ComputeError, ComputeTarget, ComputeTask};
pub use geometry::LaunchGeometry;
pub use pipeline::{Pipeline, PipelineStage};
pub use linalg::PcaModel;
pub use reduction::{min_max_decimate, min_max_decimate_into};
