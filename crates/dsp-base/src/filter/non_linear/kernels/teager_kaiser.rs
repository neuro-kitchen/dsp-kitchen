use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// CubeCL kernel for the discrete Teager-Kaiser Energy Operator (TKEO):
/// `psi[n] = x[n]^2 - x[n-1] * x[n+1]`
/// Accentuates high-frequency instantaneous energy transitions and transient bursts.
#[cube(launch)]
pub fn teager_kaiser_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = sample_position();
    let channel_idx = channel_position();

    if sample_idx < num_samples && channel_idx < num_channels {
        let channel_offset = channel_idx * num_samples;
        let linear_idx = channel_offset + sample_idx;

        if sample_idx == 0u32 || sample_idx == num_samples - 1u32 {
            let val = input[linear_idx as usize];
            output[linear_idx as usize] = val * val;
        } else {
            let curr = input[linear_idx as usize];
            let prev = input[(linear_idx - 1u32) as usize];
            let next = input[(linear_idx + 1u32) as usize];

            let energy = curr * curr - prev * next;
            // Floor negative noise artifacts to zero
            output[linear_idx as usize] = f32::max(energy, 0.0f32);
        }
    }
}
