use cubecl::prelude::*;

/// CubeCL kernel for linear vector scaling and offset: `y = x * scale + offset`.
/// Used for ADC calibration ($int16 \to \mu V$) and channel gain adjustments.
#[cube(launch)]
pub fn scale_samples_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    scale_factor: f32,
    offset: f32,
) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS as usize] = input[ABSOLUTE_POS as usize] * scale_factor + offset;
    }
}
