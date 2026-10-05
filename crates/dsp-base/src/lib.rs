//! Classical deterministic DSP, linear algebra, and filtering algorithms powered by CubeCL kernels.

#[cfg(test)]
#[macro_use]
mod test_support;

pub mod core;
pub mod filter;
pub mod spatial;
pub mod math;
pub mod linalg;
pub mod pipeline;
pub mod resampler;

pub use pipeline::{Pipeline, PipelineStage};
pub use linalg::{FastIcaModel, IcaContrast, PcaModel, PpcaModel};
pub use spatial::{SpatialWhitening, SurfaceLaplacian};
