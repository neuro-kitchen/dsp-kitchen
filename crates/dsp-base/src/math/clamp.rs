use cubecl::prelude::*;
use crate::geometry::LaunchGeometry;
use super::kernels::clamp_samples_kernel;

/// High-level host dispatcher for elementwise sample clamping.
pub fn execute_clamp<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    total_elements: usize,
    min_val: f32,
    max_val: f32,
) {
    let geom = LaunchGeometry::elementwise(client, total_elements);

    unsafe {
        clamp_samples_kernel::launch::<R>(
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
