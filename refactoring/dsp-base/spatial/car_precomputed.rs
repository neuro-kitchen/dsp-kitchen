// Parked from crates/dsp-base/src/spatial/car.rs and spatial/kernels/car.rs (no users).
// CAR with a precomputed average trace; `execute_direct_car` (single pass) is the one in use.

/// Common Average Referencing (CAR) subtraction with a precomputed average trace (`[samples]`).
#[cube(launch)]
pub fn subtract_common_average_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    common_average: &Array<F>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = sample_position();
    let channel_idx = channel_position();

    if sample_idx < num_samples && channel_idx < num_channels {
        let linear_idx = (channel_idx * num_samples + sample_idx) as usize;
        output[linear_idx] = input[linear_idx] - common_average[sample_idx as usize];
    }
}


/// High-level host dispatcher for Common Average Referencing (CAR) with precomputed average.
pub fn execute_car<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    common_average: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    let total_elements = channels * samples;

    unsafe {
        subtract_common_average_kernel::launch::<F, R>(
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

