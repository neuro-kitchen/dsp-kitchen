use cubecl::prelude::*;

/// CubeCL kernel for Second-Order Section (Biquad / IIR) filtering.
/// Executes Direct-Form II Transposed structure across each channel.
/// Parallelized across channels (`ABSOLUTE_POS_X`), iterating sequentially through time.
///
/// Coefficients: `b0, b1, b2, a1, a2` (normalized such that a0 = 1.0).
#[cube(launch)]
pub fn biquad_filter_kernel(
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

/// CubeCL kernel for 4th-order cascaded Biquad filtering (2 cascaded SOS sections in a single pass).
/// Eliminates intermediate buffer round-trips when running high-order bandpass filters.
#[cube(launch)]
pub fn cascaded_biquad_4th_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    // Section 1
    s1_b0: f32,
    s1_b1: f32,
    s1_b2: f32,
    s1_a1: f32,
    s1_a2: f32,
    // Section 2
    s2_b0: f32,
    s2_b1: f32,
    s2_b2: f32,
    s2_a1: f32,
    s2_a2: f32,
    num_channels: u32,
    num_samples: u32,
) {
    let channel_idx = ABSOLUTE_POS_X;

    if channel_idx < num_channels {
        let mut sec1_s1 = 0.0f32;
        let mut sec1_s2 = 0.0f32;
        let mut sec2_s1 = 0.0f32;
        let mut sec2_s2 = 0.0f32;

        let channel_offset = channel_idx * num_samples;
        let mut s = 0u32;

        while s < num_samples {
            let idx = (channel_offset + s) as usize;
            let x = input[idx];

            // Section 1 Direct-Form II Transposed
            let y1 = s1_b0 * x + sec1_s1;
            sec1_s1 = s1_b1 * x - s1_a1 * y1 + sec1_s2;
            sec1_s2 = s1_b2 * x - s1_a2 * y1;

            // Section 2 Direct-Form II Transposed fed directly from Section 1
            let y2 = s2_b0 * y1 + sec2_s1;
            sec2_s1 = s2_b1 * y1 - s2_a1 * y2 + sec2_s2;
            sec2_s2 = s2_b2 * y1 - s2_a2 * y2;

            output[idx] = y2;
            s = s + 1u32;
        }
    }
}
