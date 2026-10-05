use cubecl::prelude::*;

/// Sample clamping / rectification: `y = clamp(x, min, max)`.
#[cube(launch)]
pub fn clamp_samples_kernel<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>, min_val: F, max_val: F) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = F::clamp(input[ABSOLUTE_POS], min_val, max_val);
    }
}
