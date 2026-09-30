//! Out-of-core streaming spike sorting, dynamic halo scheduling, and online Welford template accumulation.

pub mod accumulator;
pub mod config;
pub mod runner;

pub use accumulator::TemplateAccumulator;
pub use config::{SINC_RESAMPLE_MARGIN, StreamingSortConfig};
pub use runner::{StreamingSortResult, StreamingSpikeRunner, calibrate_noise};
