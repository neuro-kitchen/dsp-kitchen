use cubecl::prelude::*;
use cubecl::server::Handle;
use dspk_core::compute::LaunchGeometry;

use super::kernels::scale_samples_kernel;
use crate::core::DspFloat;

/// `output = input · scale + offset` over `len` values (any shape: element-wise).
pub fn execute_scaling<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    len: usize,
    scale: F,
    offset: F,
) {
    let geom = LaunchGeometry::elementwise(client, len);
    unsafe {
        scale_samples_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), len),
            BufferArg::from_raw_parts(output.clone(), len),
            scale,
            offset,
        );
    }
}
