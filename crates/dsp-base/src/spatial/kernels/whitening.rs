use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// `[C, C] × [C, S]` spatial projection (whitening, Laplacian): `output[o, t] = Σ_i W[o, i] · x[i, t]`.
/// One unit per `(output channel, sample)` ([`dsp_core::compute::LaunchGeometry::channels_samples`]).
#[cube(launch)]
pub fn spatial_matrix_multiply_kernel<F: Float>(
    input: &Array<F>,
    weights: &Array<F>,
    output: &mut Array<F>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = sample_position();
    let out_ch = channel_position();

    if sample_idx < num_samples && out_ch < num_channels {
        let row_offset = out_ch * num_channels;
        let mut acc = F::new(0.0f32);
        let mut in_ch = 0u32;
        while in_ch < num_channels {
            acc += weights[(row_offset + in_ch) as usize] * input[(in_ch * num_samples + sample_idx) as usize];
            in_ch += 1u32;
        }
        output[(out_ch * num_samples + sample_idx) as usize] = acc;
    }
}
