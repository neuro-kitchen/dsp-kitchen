use cubecl::prelude::*;
use crate::geometry::LaunchGeometry;
use super::kernels::neo_kernel;

/// High-level host dispatcher for the discrete Nonlinear Energy Operator.
pub fn execute_neo<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    is_cpu: bool,
) {
    let geom = LaunchGeometry::for_channels_and_samples(channels, samples, is_cpu);
    let total_elements = channels * samples;

    unsafe {
        neo_kernel::launch::<R>(
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
