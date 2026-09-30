use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;
use super::kernels::teager_kaiser_kernel;

/// High-level host dispatcher for the discrete Teager-Kaiser Energy Operator.
pub fn execute_teager_kaiser<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    let total_elements = channels * samples;

    unsafe {
        teager_kaiser_kernel::launch::<R>(
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
