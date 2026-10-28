use cubecl::prelude::*;
use dspk_core::compute::element_position;

/// Sample clamping / rectification: `y = clamp(x, min, max)`.
#[cube(launch)]
pub fn clamp_samples_kernel<F: Float + CubeElement + LaunchArg>(input: &[F], output: &mut [F], min_val: F, max_val: F) {
    let i = element_position();
    if i < input.len() {
        output[i] = F::clamp(input[i], min_val, max_val);
    }
}
