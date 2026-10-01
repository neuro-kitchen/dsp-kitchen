//! Out-of-core streaming spike sorting, dynamic halo scheduling, and online Welford template accumulation.

pub mod accumulator;
pub mod config;
pub mod kernels;
pub mod runner;

pub use accumulator::TemplateAccumulator;
pub use config::{SINC_RESAMPLE_MARGIN, StreamingSortConfig};
pub use kernels::{
    BatchTemplateStats, execute_reduce_templates_in_vram, reduce_channel_templates_kernel,
};
pub use runner::{StreamingSortResult, StreamingSpikeRunner, calibrate_noise};
