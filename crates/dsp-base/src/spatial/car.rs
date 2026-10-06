use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;

use crate::core::DspFloat;
pub use super::kernels::direct_car_kernel;

/// High-level host dispatcher for direct, single-pass Common Average Referencing (CAR).
/// Computes and subtracts the common average directly in VRAM without auxiliary buffers.
pub fn execute_direct_car<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    let geom = LaunchGeometry::per_sample(client, samples);
    let total_elements = channels * samples;

    unsafe {
        direct_car_kernel::launch::<F, R>(
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

/// Zero-parameter spatial stage descriptor for Common Average Referencing across channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CommonAverageReference;

impl From<CommonAverageReference> for crate::pipeline::PipelineStage {
    fn from(_: CommonAverageReference) -> Self {
        crate::pipeline::PipelineStage::CommonAverageReference
    }
}
