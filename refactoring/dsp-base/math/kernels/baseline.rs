use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// Per-channel baseline subtraction (DC removal) on a channel-major `[channels, samples]` buffer.
#[cube(launch)]
pub fn baseline_subtract_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    channel_baselines: &Array<F>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = sample_position();
    let channel_idx = channel_position();

    if sample_idx < num_samples && channel_idx < num_channels {
        let linear_idx = (channel_idx * num_samples + sample_idx) as usize;
        output[linear_idx] = input[linear_idx] - channel_baselines[channel_idx as usize];
    }
}
