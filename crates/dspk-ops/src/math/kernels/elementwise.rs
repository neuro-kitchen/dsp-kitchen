use cubecl::prelude::*;
use dspk_core::compute::element_position;

/// One function of each sample (`op`, fixed at compile time): `0` `|x|`, `1` `x²`, `2` `√max(x, 0)`,
/// `3` `log₁₀ max(|x|, floor)`.
#[cube(launch)]
pub fn elementwise_kernel<F: Float + CubeElement + LaunchArg>(input: &[F], output: &mut [F], floor: F, #[comptime] op: u32) {
    let i = element_position();
    if i < input.len() {
        let x = input[i];
        if comptime!(op == 0) {
            output[i] = F::abs(x);
        } else if comptime!(op == 1) {
            output[i] = x * x;
        } else if comptime!(op == 2) {
            output[i] = F::sqrt(F::max(x, F::new(0.0f32)));
        } else {
            output[i] = F::ln(F::max(F::abs(x), floor)) * F::new(core::f32::consts::LOG10_E);
        }
    }
}
