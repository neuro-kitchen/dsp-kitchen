use cubecl::prelude::*;
use crate::geometry::LaunchGeometry;
use super::kernels::baseline_subtract_kernel;

/// High-level host dispatcher for subtracting per-channel baseline offsets.
pub fn execute_baseline_subtract<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channel_baselines: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    let total_elements = channels * samples;

    unsafe {
        baseline_subtract_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total_elements),
            ArrayArg::from_raw_parts(output.clone(), total_elements),
            ArrayArg::from_raw_parts(channel_baselines.clone(), channels),
            channels as u32,
            samples as u32,
        );
    }
}
