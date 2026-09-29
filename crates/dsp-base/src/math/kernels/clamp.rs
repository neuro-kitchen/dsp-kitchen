use cubecl::prelude::*;

/// CubeCL kernel for sample clamping / rectification: `y = clamp(x, min, max)`.
#[cube(launch)]
pub fn clamp_samples_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    min_val: f32,
    max_val: f32,
) {
    if ABSOLUTE_POS < input.len() {
        let val = input[ABSOLUTE_POS as usize];
        output[ABSOLUTE_POS as usize] = f32::clamp(val, min_val, max_val);
    }
}
