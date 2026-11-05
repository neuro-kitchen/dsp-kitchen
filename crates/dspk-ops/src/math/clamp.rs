use cubecl::prelude::*;
use cubecl::server::Handle;
use dspk_core::compute::LaunchGeometry;

use super::kernels::clamp_samples_kernel;
use crate::core::DspFloat;

/// `output = clamp(input, min, max)` over `len` values (any shape: element-wise).
pub fn execute_clamp<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    len: usize,
    min: F,
    max: F,
) {
    let geom = LaunchGeometry::elementwise(client, len);
    unsafe {
        clamp_samples_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), len),
            BufferArg::from_raw_parts(output.clone(), len),
            min,
            max,
        );
    }
}
