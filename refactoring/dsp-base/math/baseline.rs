use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;

use super::kernels::baseline_subtract_kernel;
use crate::core::DspFloat;

/// Subtracts `channel_baselines[c]` (a device buffer of `channels` values) from every sample of
/// channel `c` of a `[channels, samples]` buffer.
pub fn execute_baseline_subtract<R: Runtime, F: DspFloat>(
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
        baseline_subtract_kernel::launch::<F, R>(
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
