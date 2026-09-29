use cubecl::prelude::*;

/// CubeCL kernel for subtracting per-channel baseline offsets (DC bias removal).
/// Assumes Channel-Major order: [num_channels, num_samples].
#[cube(launch)]
pub fn baseline_subtract_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    channel_baselines: &Array<f32>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = ABSOLUTE_POS_X;
    let channel_idx = ABSOLUTE_POS_Y;

    if sample_idx < num_samples && channel_idx < num_channels {
        let linear_idx = channel_idx * num_samples + sample_idx;
        let baseline = channel_baselines[channel_idx as usize];
        output[linear_idx as usize] = input[linear_idx as usize] - baseline;
    }
}
