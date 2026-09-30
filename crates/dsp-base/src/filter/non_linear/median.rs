use cubecl::prelude::*;
use crate::geometry::LaunchGeometry;
use super::kernels::median_filter_9p_kernel;

/// High-level host dispatcher for 9-point temporal median filter across all channels.
pub fn execute_median_9p<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    let total_elements = channels * samples;

    unsafe {
        median_filter_9p_kernel::launch::<R>(
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
