use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;

use super::kernels::scale_samples_kernel;
use crate::core::DspFloat;

/// `output = input · scale_factor + offset` over `total_elements` values.
pub fn execute_scaling<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    total_elements: usize,
    scale_factor: F,
    offset: F,
) {
    let geom = LaunchGeometry::elementwise(client, total_elements);
    unsafe {
        scale_samples_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), total_elements),
            BufferArg::from_raw_parts(output.clone(), total_elements),
            scale_factor,
            offset,
        );
    }
}
