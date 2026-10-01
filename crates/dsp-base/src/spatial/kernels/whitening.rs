use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// CubeCL kernel for `[C, C] x [C, S]` spatial linear projection (ZCA/local whitening or Laplacian).
/// Parallelized across `(channel, sample)` via [`dsp_core::compute::LaunchGeometry::channels_samples`].
#[cube(launch)]
pub fn spatial_matrix_multiply_kernel(
    input: &Array<f32>,
    weights: &Array<f32>,
    output: &mut Array<f32>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = sample_position();
    let out_ch = channel_position();

    if sample_idx < num_samples && out_ch < num_channels {
        let row_offset = out_ch * num_channels;
        let mut acc = 0.0f32;
        let mut in_ch = 0u32;

        while in_ch < num_channels {
            let w = weights[(row_offset + in_ch) as usize];
            let x = input[(in_ch * num_samples + sample_idx) as usize];
            acc = acc + w * x;
            in_ch = in_ch + 1u32;
        }

        output[(out_ch * num_samples + sample_idx) as usize] = acc;
    }
}
