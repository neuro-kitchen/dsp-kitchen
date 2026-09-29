use cubecl::prelude::*;

/// CubeCL kernel for a specialized narrow-band power-line notch filter (e.g. 50/60 Hz).
/// Direct-Form II Transposed execution across channels.
#[cube(launch)]
pub fn notch_filter_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    num_channels: u32,
    num_samples: u32,
) {
    let channel_idx = ABSOLUTE_POS_X;

    if channel_idx < num_channels {
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;

        let channel_offset = channel_idx * num_samples;
        let mut s = 0u32;

        while s < num_samples {
            let idx = (channel_offset + s) as usize;
            let x = input[idx];
            let y = b0 * x + s1;

            s1 = b1 * x - a1 * y + s2;
            s2 = b2 * x - a2 * y;

            output[idx] = y;
            s = s + 1u32;
        }
    }
}
