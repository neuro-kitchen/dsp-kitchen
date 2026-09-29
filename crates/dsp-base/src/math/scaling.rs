use cubecl::prelude::*;
use crate::geometry::LaunchGeometry;
use super::kernels::scale_samples_kernel;

/// High-level host dispatcher for scaling and offsetting a signal buffer.
pub fn execute_scaling<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    total_elements: usize,
    scale_factor: f32,
    offset: f32,
    is_cpu: bool,
) {
    let geom = LaunchGeometry::for_1d(total_elements, is_cpu);

    unsafe {
        scale_samples_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total_elements),
            ArrayArg::from_raw_parts(output.clone(), total_elements),
            scale_factor,
            offset,
        );
    }
}
