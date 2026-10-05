use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

use crate::core::read_extended;

/// Median of three with three min/max operations.
#[cube]
fn med3<F: Float>(a: F, b: F, c: F) -> F {
    F::max(F::min(a, b), F::min(F::max(a, b), c))
}

/// Exact median of nine values from min/max operations only (no branches, no sorting): sort each
/// column of three, then take the median of the largest minimum, the median of medians and the
/// smallest maximum. Exact for every ordering of nine values (checked over all 9! permutations).
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn med9<F: Float>(p0: F, p1: F, p2: F, p3: F, p4: F, p5: F, p6: F, p7: F, p8: F) -> F {
    let max_min = F::max(F::min(p0, F::min(p1, p2)), F::max(F::min(p3, F::min(p4, p5)), F::min(p6, F::min(p7, p8))));
    let min_max = F::min(F::max(p0, F::max(p1, p2)), F::min(F::max(p3, F::max(p4, p5)), F::max(p6, F::max(p7, p8))));
    let mid_mid = med3::<F>(med3::<F>(p0, p1, p2), med3::<F>(p3, p4, p5), med3::<F>(p6, p7, p8));
    med3::<F>(max_min, mid_mid, min_max)
}

/// Median of `window[0..width]` by rank: the value with at most `width / 2` smaller values and more
/// than `width / 2` values smaller or equal (ties included). `width` is comptime, so the window stays
/// in registers.
#[cube]
fn median_by_rank<F: Float>(window: &Array<F>, #[comptime] width: u32) -> F {
    let half = comptime!(width / 2);
    let mut median = window[0usize];
    #[unroll]
    for i in 0..width {
        let x = window[i as usize];
        let mut less: u32 = 0u32;
        let mut less_equal: u32 = 0u32;
        #[unroll]
        for j in 0..width {
            let y = window[j as usize];
            if y < x {
                less += 1u32;
            }
            if y <= x {
                less_equal += 1u32;
            }
        }
        if less <= half && less_equal > half {
            median = x;
        }
    }
    median
}

/// Running median over `width` samples (odd, comptime) centred on each sample of a channel-major
/// `[channels, samples]` buffer. Samples beyond either end come from `edge` (an `EdgeMode` id; scipy
/// `signal.medfilt` reads zeros). Width 9 uses [`med9`]; other widths rank the window.
#[cube(launch)]
pub fn median_filter_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    num_channels: u32,
    num_samples: u32,
    #[comptime] width: u32,
    #[comptime] edge: u32,
) {
    let t = sample_position();
    let ch = channel_position();

    if t < num_samples && ch < num_channels {
        let radius = comptime!(width / 2);
        let base = (ch * num_samples) as usize;
        let mut window = Array::<F>::new(comptime!(width as usize));
        if t >= radius && t + radius < num_samples {
            let first = base + (t - radius) as usize;
            #[unroll]
            for k in 0..width {
                window[k as usize] = input[first + k as usize];
            }
        } else {
            // Logical sample t + k − radius is extended position t + k
            #[unroll]
            for k in 0..width {
                window[k as usize] = read_extended::<F>(input, base, num_samples, radius, t + k, edge);
            }
        }
        let median = if comptime!(width == 9) {
            med9::<F>(window[0], window[1], window[2], window[3], window[4], window[5], window[6], window[7], window[8])
        } else {
            median_by_rank::<F>(&window, width)
        };
        output[base + t as usize] = median;
    }
}
