//! Classical deterministic DSP, linear algebra, and filtering algorithms powered by CubeCL kernels.

pub mod filter;
pub mod spatial;
pub mod math;
pub mod linalg;
pub mod pipeline;
pub mod resampler;

pub use pipeline::{Pipeline, PipelineStage};
pub use linalg::{FastIcaModel, IcaContrast, PcaModel, PpcaModel};
pub use spatial::{SpatialWhitening, SurfaceLaplacian};
pub use resampler::{min_max_decimate, min_max_decimate_into};
