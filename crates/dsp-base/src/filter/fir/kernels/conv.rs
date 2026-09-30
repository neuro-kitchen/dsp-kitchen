use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// CubeCL kernel for direct-form multi-channel FIR convolution.
/// Assumes Channel-Major order: [num_channels, num_samples].
/// Parallelized across both samples (X) and channels (Y).
#[cube(launch)]
pub fn fir_filter_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    taps: &Array<f32>,
    num_channels: u32,
    num_samples: u32,
    num_taps: u32,
) {
    let sample_idx = sample_position();
    let channel_idx = channel_position();

    if sample_idx < num_samples && channel_idx < num_channels {
        let channel_offset = channel_idx * num_samples;
        let mut sum = 0.0f32;
        let mut tap_idx = 0u32;

        while tap_idx < num_taps {
            if sample_idx >= tap_idx {
                let in_idx = channel_offset + (sample_idx - tap_idx);
                sum = sum + input[in_idx as usize] * taps[tap_idx as usize];
            }
            tap_idx = tap_idx + 1u32;
        }

        output[(channel_offset + sample_idx) as usize] = sum;
    }
}
