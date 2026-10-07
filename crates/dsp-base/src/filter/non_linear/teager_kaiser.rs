use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;

use super::kernels::teager_kaiser_kernel;
use crate::core::{DspFloat, EdgeMode};

/// Edge handling of the Teager-Kaiser operator (no scipy equivalent; mirrors the neighbours).
pub const TEAGER_KAISER_DEFAULT_EDGE: EdgeMode = EdgeMode::Reflect;

/// Discrete Teager-Kaiser energy operator along time of a `[channels, samples]` buffer.
pub fn execute_teager_kaiser<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    edge: EdgeMode,
) {
    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    let total_elements = channels * samples;
    unsafe {
        teager_kaiser_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), total_elements),
            BufferArg::from_raw_parts(output.clone(), total_elements),
            channels as u32,
            samples as u32,
            edge.id(),
        );
    }
}
