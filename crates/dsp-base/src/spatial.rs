use cubecl::prelude::*;
use crate::geometry::LaunchGeometry;

/// Common Average Referencing (CAR) subtraction with precomputed average trace.
#[cube(launch)]
pub fn subtract_common_average_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    common_average: &Array<f32>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = ABSOLUTE_POS_X;
    let channel_idx = ABSOLUTE_POS_Y;

    if sample_idx < num_samples && channel_idx < num_channels {
        let linear_idx = channel_idx * num_samples + sample_idx;
        let avg = common_average[sample_idx as usize];
        output[linear_idx as usize] = input[linear_idx as usize] - avg;
    }
}

/// Direct Common Average Referencing (CAR) kernel.
/// Parallelized across samples (`ABSOLUTE_POS_X`).
/// Computes the mean across channels at each time step and subtracts it directly in one pass.
/// Requires zero intermediate buffer allocation!
#[cube(launch)]
pub fn direct_car_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = ABSOLUTE_POS_X;

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

/// High-level host dispatcher for Common Average Referencing (CAR) with precomputed average.
pub fn execute_car<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    common_average: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    is_cpu: bool,
) {
    let geom = LaunchGeometry::for_channels_and_samples(channels, samples, is_cpu);
    let total_elements = channels * samples;

    unsafe {
        subtract_common_average_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total_elements),
            ArrayArg::from_raw_parts(output.clone(), total_elements),
            ArrayArg::from_raw_parts(common_average.clone(), samples),
            channels as u32,
            samples as u32,
        );
    }
}

/// High-level host dispatcher for direct, single-pass Common Average Referencing (CAR).
/// Computes and subtracts the common average directly in VRAM without auxiliary buffers.
pub fn execute_direct_car<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    is_cpu: bool,
) {
    let geom = LaunchGeometry::for_1d(samples, is_cpu);
    let total_elements = channels * samples;

    unsafe {
        direct_car_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total_elements),
            ArrayArg::from_raw_parts(output.clone(), total_elements),
            channels as u32,
            samples as u32,
        );
    }
}
