use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;

use super::kernels::clamp_samples_kernel;
use crate::core::DspFloat;

/// `output = clamp(input, min_val, max_val)` over `total_elements` values.
pub fn execute_clamp<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    total_elements: usize,
    min_val: F,
    max_val: F,
) {
    let geom = LaunchGeometry::elementwise(client, total_elements);
    unsafe {
        clamp_samples_kernel::launch::<F, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total_elements),
            ArrayArg::from_raw_parts(output.clone(), total_elements),
            min_val,
            max_val,
        );
    }
}
