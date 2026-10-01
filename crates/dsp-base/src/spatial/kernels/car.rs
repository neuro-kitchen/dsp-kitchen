use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// Common Average Referencing (CAR) subtraction with precomputed average trace.
#[cube(launch)]
pub fn subtract_common_average_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    common_average: &Array<f32>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = sample_position();
    let channel_idx = channel_position();

    if sample_idx < num_samples && channel_idx < num_channels {
        let linear_idx = channel_idx * num_samples + sample_idx;
        let avg = common_average[sample_idx as usize];
        output[linear_idx as usize] = input[linear_idx as usize] - avg;
    }
}

/// Direct Common Average Referencing (CAR) kernel.
/// Parallelized across samples ([`sample_position`]).
/// Computes the mean across channels at each time step and subtracts it directly in one pass.
/// Requires zero intermediate buffer allocation!
#[cube(launch)]
pub fn direct_car_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = sample_position();

    if sample_idx < num_samples {
        let mut sum = 0.0f32;
        let mut c = 0u32;

        // Step 1: Compute average across all channels at this time sample
        while c < num_channels {
            let idx = (c * num_samples + sample_idx) as usize;
            sum = sum + input[idx];
            c = c + 1u32;
        }

        let avg = sum / (num_channels as f32);

        // Step 2: Subtract the average
        c = 0u32;
        while c < num_channels {
            let idx = (c * num_samples + sample_idx) as usize;
            output[idx] = input[idx] - avg;
            c = c + 1u32;
        }
    }
}
