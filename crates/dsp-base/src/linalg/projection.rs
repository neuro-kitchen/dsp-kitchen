//! Linear projections `Y = Wᵀ (X − mean)` with their weights uploaded once.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::kernels::pca_project_kernel;
use crate::core::{buffer, cast_f32, DspFloat};

/// Projection weights `[channels, components]` and channel means on the device (from
/// [`crate::linalg::PcaModel::to_device`] or [`crate::linalg::PpcaModel::to_device`]), reused across
/// calls.
#[derive(Debug, Clone)]
pub struct DeviceProjection {
    weights: Handle,
    mean: Handle,
    channels: usize,
    components: usize,
}

impl DeviceProjection {
    /// Uploads row-major `weights` (`[channels, components]`) and `mean` (`channels`) as `F`.
    pub fn upload<R: Runtime, F: DspFloat>(client: &ComputeClient<R>, weights: &[f32], mean: &[f32], channels: usize, components: usize) -> Self {
        assert_eq!(weights.len(), channels * components, "weights size mismatch");
        assert_eq!(mean.len(), channels, "mean size mismatch");
        Self {
            weights: buffer::upload(client, &cast_f32::<F>(weights)),
            mean: buffer::upload(client, &cast_f32::<F>(mean)),
            channels,
            components,
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn components(&self) -> usize {
        self.components
    }

    /// Projects a `[channels, samples]` buffer of `F` into `output` (`[components, samples]`).
    pub fn project<R: Runtime, F: DspFloat>(&self, client: &ComputeClient<R>, input: &Handle, output: &Handle, samples: usize) {
        let geom = LaunchGeometry::channels_samples(client, self.components, samples);
        unsafe {
            pca_project_kernel::launch::<F, R>(
                client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(input.clone(), self.channels * samples),
                ArrayArg::from_raw_parts(self.weights.clone(), self.channels * self.components),
                ArrayArg::from_raw_parts(self.mean.clone(), self.channels),
                ArrayArg::from_raw_parts(output.clone(), self.components * samples),
                self.channels as u32,
                samples as u32,
                self.components as u32,
            );
        }
    }
}
