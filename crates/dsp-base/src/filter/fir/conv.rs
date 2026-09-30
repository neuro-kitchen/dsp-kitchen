use cubecl::prelude::*;
use crate::geometry::LaunchGeometry;
use super::kernels::fir_filter_kernel;

/// High-level host dispatcher for multi-channel temporal FIR filtering.
pub fn execute_fir<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    taps: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    num_taps: usize,
) {
    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    let total_elements = channels * samples;

    unsafe {
        fir_filter_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total_elements),
            ArrayArg::from_raw_parts(output.clone(), total_elements),
            ArrayArg::from_raw_parts(taps.clone(), num_taps),
            channels as u32,
            samples as u32,
            num_taps as u32,
        );
    }
}
