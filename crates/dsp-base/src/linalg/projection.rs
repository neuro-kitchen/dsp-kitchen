//! Linear projections `Y = Wᵀ (X − mean)` with their weights uploaded once: the input is centred
//! into a reused scratch buffer, then projected with one matrix product ([`fn@super::matmul`]).
//! Centring first (rather than subtracting `Wᵀ mean` afterwards) keeps a large mean from cancelling
//! the signal's digits.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::kernels::center_rows_kernel;
use super::matmul::{matmul, MatrixView};
use crate::core::{buffer, cast_f32, DspFloat, Scratch};

/// Projection weights `[channels, components]` and channel means on the device (from
/// [`crate::linalg::PcaModel::to_device`] or [`crate::linalg::PpcaModel::to_device`]), reused across
/// calls.
#[derive(Debug, Clone)]
pub struct DeviceProjection {
    weights: Handle,
    mean: Handle,
    channels: usize,
    components: usize,
    centred: Scratch,
}

impl DeviceProjection {
    /// Uploads row-major `weights` (`[channels, components]`) and `mean` (`channels`) as `F`.
    pub fn upload<F: DspFloat>(client: &Client, weights: &[f32], mean: &[f32], channels: usize, components: usize) -> Self {
        assert_eq!(weights.len(), channels * components, "weights size mismatch");
        assert_eq!(mean.len(), channels, "mean size mismatch");
        Self {
            weights: buffer::upload(client, &cast_f32::<F>(weights)),
            mean: buffer::upload(client, &cast_f32::<F>(mean)),
            channels,
            components,
            centred: Scratch::new(),
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn components(&self) -> usize {
        self.components
    }

    /// Projects a `[channels, samples]` buffer of `F` into `output` (`[components, samples]`).
    pub fn project<F: DspFloat>(&mut self, client: &Client, input: &Handle, output: &Handle, samples: usize) {
        let total = self.channels * samples;
        if total == 0 || self.components == 0 {
            return;
        }
        let centred = self.centred.get::<F>(client, total);
        let geom = LaunchGeometry::elementwise(client, total);
        // SAFETY: `input` and `centred` hold `total`, `mean` `channels` values of `F`
        unsafe {
            center_rows_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(input.clone(), total),
                BufferArg::from_raw_parts(self.mean.clone(), self.channels),
                BufferArg::from_raw_parts(centred.clone(), total),
                0,
                samples as u32,
                samples as u32,
                total as u32,
            );
        }
        let weights = MatrixView::row_major(&self.weights, self.channels * self.components, self.channels, self.components);
        let x = MatrixView::row_major(&centred, total, self.channels, samples);
        matmul::<F>(client, &weights.transposed(), &x, output, self.components * samples);
    }
}
