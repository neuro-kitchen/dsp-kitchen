//! Out-of-core streaming spike detection with per-channel templates, dynamic halo scheduling, and
//! online template accumulation.

pub mod accumulator;
pub mod config;
pub mod kernels;
pub mod runner;

pub use accumulator::TemplateAccumulator;
pub use config::{SINC_RESAMPLE_MARGIN, StreamingDetectionConfig};
pub use kernels::{
    BatchTemplateStats, execute_reduce_templates_in_vram, reduce_channel_templates_kernel,
};
pub use runner::{StreamingDetectionResult, StreamingDetector, calibrate_noise};
