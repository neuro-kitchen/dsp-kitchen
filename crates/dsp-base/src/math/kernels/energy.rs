use cubecl::prelude::*;

/// CubeCL kernel for the discrete Nonlinear Energy Operator (NEO):
/// `psi[n] = x[n]^2 - x[n-1] * x[n+1]`
/// Accentuates action potentials and transient energy bursts in neural traces.
#[cube(launch)]
pub fn neo_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = ABSOLUTE_POS_X;
    let channel_idx = ABSOLUTE_POS_Y;

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
