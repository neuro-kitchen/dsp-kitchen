//! Signal processing for multi-channel recordings on the device (any GPU through wgpu, CUDA or HIP,
//! or the CPU), written as CubeCL kernels: filters, spatial operators, statistics, linear algebra,
//! peak finding and resampling.
//!
//! # Where things are
//!
//! | Need | Module | Start with |
//! |---|---|---|
//! | Chain preprocessing steps | [`pipeline`] | [`Pipeline`], [`PipelineStage`] |
//! | Filters (IIR, FIR, median, templates) | [`filter`] | [`filter::FilterSpec`], [`filter::DeviceFilter`] |
//! | Mix channels (CAR, whitening, Laplacian) | [`spatial`] | [`SpatialWhitening`], [`SurfaceLaplacian`] |
//! | Noise, percentiles, element-wise math | [`math`] | [`math::estimate_noise_std`] |
//! | Matrix products, PCA, ICA | [`linalg`] | [`PcaModel`], [`linalg::matmul`](fn@linalg::matmul) |
//! | Peaks (`find_peaks`) | [`peaks`] | [`peaks::find_peak_candidates`] |
//! | Resampling | [`resampler`] | [`resampler::resample_poly`] |
//! | Device buffers and shared kernels | [`core`] | [`core::buffer`] |
//!
//! Data is a `[channels, samples]` buffer, channel-major. Functions take a cubecl `Client`, from
//! [`dsp_core::compute::ComputeTarget`]: `ComputeTarget::from_env()?.client()?` picks the runtime
//! named by `DSP_KITCHEN_RUNTIME`, otherwise the first compiled in. See [`Pipeline`] for a complete
//! example.

#[cfg(test)]
#[macro_use]
mod test_support;

pub mod core;
pub mod filter;
pub mod spatial;
pub mod math;
pub mod linalg;
pub mod peaks;
pub mod pipeline;
pub mod resampler;

pub use pipeline::{Pipeline, PipelineStage};
pub use linalg::{FastIcaModel, IcaContrast, PcaModel, PpcaModel};
pub use spatial::{SpatialWhitening, SurfaceLaplacian};
