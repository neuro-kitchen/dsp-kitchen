use cubecl::prelude::*;
use dspk_core::compute::element_position;

/// Linear scaling and offset: `y = x · scale + offset` (gain calibration, unit conversion).
#[cube(launch)]
pub fn scale_samples_kernel<F: Float + CubeElement + LaunchArg>(input: &[F], output: &mut [F], scale_factor: F, offset: F) {
    let i = element_position();
    if i < input.len() {
        output[i] = input[i] * scale_factor + offset;
    }
}
