use cubecl::prelude::*;

/// Linear scaling and offset: `y = x · scale + offset` (gain calibration, unit conversion).
#[cube(launch)]
pub fn scale_samples_kernel<F: Float + CubeElement + LaunchArg>(input: &[F], output: &mut [F], scale_factor: F, offset: F) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * scale_factor + offset;
    }
}
