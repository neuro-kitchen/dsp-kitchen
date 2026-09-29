use cubecl::prelude::*;

/// 3-element median using 3 min/max comparisons.
#[cube]
fn med3(a: f32, b: f32, c: f32) -> f32 {
    let min_ab = f32::min(a, b);
    let max_ab = f32::max(a, b);
    f32::max(min_ab, f32::min(max_ab, c))
}

/// 19-operation branchless median filter across 9 values.
/// Operates entirely in registers with zero warp divergence.
#[cube]
pub fn med9(
    p0: f32, p1: f32, p2: f32,
    p3: f32, p4: f32, p5: f32,
    p6: f32, p7: f32, p8: f32,
) -> f32 {
    // 1. Sort each 3-element column
    let min1 = f32::min(p0, f32::min(p1, p2));
    let max1 = f32::max(p0, f32::max(p1, p2));
    let mid1 = med3(p0, p1, p2);

    let min2 = f32::min(p3, f32::min(p4, p5));
    let max2 = f32::max(p3, f32::max(p4, p5));
    let mid2 = med3(p3, p4, p5);

    let min3 = f32::min(p6, f32::min(p7, p8));
    let max3 = f32::max(p6, f32::max(p7, p8));
    let mid3 = med3(p6, p7, p8);

    // 2. Cross-column reduction
    let max_min = f32::max(min1, f32::max(min2, min3));
    let min_max = f32::min(max1, f32::min(max2, max3));
    let mid_mid = med3(mid1, mid2, mid3);

    med3(max_min, mid_mid, min_max)
}

/// CubeCL kernel for 9-point temporal median filter across multi-channel signal.
#[cube(launch)]
pub fn median_filter_9p_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    num_channels: u32,
    num_samples: u32,
) {
    let sample_idx = ABSOLUTE_POS_X;
    let channel_idx = ABSOLUTE_POS_Y;

    if sample_idx < num_samples && channel_idx < num_channels {
        let channel_offset = channel_idx * num_samples;
        let linear_idx = channel_offset + sample_idx;

        // Interior fast-path (99% of samples)
        if sample_idx >= 4u32 && sample_idx + 4u32 < num_samples {
            let p0 = input[(linear_idx - 4u32) as usize];
            let p1 = input[(linear_idx - 3u32) as usize];
            let p2 = input[(linear_idx - 2u32) as usize];
            let p3 = input[(linear_idx - 1u32) as usize];
            let p4 = input[linear_idx as usize];
            let p5 = input[(linear_idx + 1u32) as usize];
            let p6 = input[(linear_idx + 2u32) as usize];
            let p7 = input[(linear_idx + 3u32) as usize];
            let p8 = input[(linear_idx + 4u32) as usize];

            output[linear_idx as usize] = med9(p0, p1, p2, p3, p4, p5, p6, p7, p8);
        } else {
            // Border pass-through
            output[linear_idx as usize] = input[linear_idx as usize];
        }
    }
}
