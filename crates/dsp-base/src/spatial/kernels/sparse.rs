use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// Spatial projection with at most `width` nonzero weights per output channel (ELLPACK rows):
/// `output[o, t] = Σ_k values[o, k] · input[indices[o, k], t]`. Padding entries carry weight 0.
/// One unit per `(output channel, sample)`.
#[cube(launch)]
pub fn sparse_rows_multiply_kernel<F: Float>(
    input: &[F],
    values: &[F],
    indices: &[u32],
    output: &mut [F],
    num_channels: u32,
    num_samples: u32,
    width: u32,
) {
    let t = sample_position();
    let out_ch = channel_position();

    if t < num_samples && out_ch < num_channels {
        let row = out_ch * width;
        let mut acc = F::new(0.0f32);
        let mut k = 0u32;
        while k < width {
            let in_ch = indices[(row + k) as usize];
            acc += values[(row + k) as usize] * input[(in_ch * num_samples + t) as usize];
            k += 1u32;
        }
        output[(out_ch * num_samples + t) as usize] = acc;
    }
}
