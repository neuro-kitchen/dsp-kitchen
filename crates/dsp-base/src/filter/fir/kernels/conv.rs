use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// CubeCL kernel for direct-form causal multi-channel FIR convolution.
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

/// CubeCL kernel for zero-phase symmetric centered FIR convolution of length `2 * radius + 1`.
/// Boundary samples clamp to `[0, num_samples - 1]`.
#[cube(launch)]
pub fn fir_centered_filter_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    taps: &Array<f32>,
    num_channels: u32,
    num_samples: u32,
    radius: u32,
) {
    let sample_idx = sample_position();
    let channel_idx = channel_position();

    if sample_idx < num_samples && channel_idx < num_channels {
        let channel_offset = channel_idx * num_samples;
        let num_taps = 2u32 * radius + 1u32;
        let last_sample = num_samples - 1u32;
        let mut sum = 0.0f32;
        let mut tap_idx = 0u32;

        while tap_idx < num_taps {
            let shifted = sample_idx + tap_idx;
            let mut src_sample = 0u32;
            if shifted >= radius {
                let s = shifted - radius;
                if s > last_sample {
                    src_sample = last_sample;
                } else {
                    src_sample = s;
                }
            }
            let in_idx = channel_offset + src_sample;
            sum = sum + input[in_idx as usize] * taps[tap_idx as usize];
            tap_idx = tap_idx + 1u32;
        }

        output[(channel_offset + sample_idx) as usize] = sum;
    }
}
