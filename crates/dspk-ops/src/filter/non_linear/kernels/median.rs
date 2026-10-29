use cubecl::prelude::*;
use dspk_core::compute::{channel_position, element_position, sample_position};

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
        let mut less = 0u32;
        let mut less_equal = 0u32;
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
/// `signal.medfilt` reads zeros). Width 9 uses [`fn@med9`]; other widths rank the window.
#[cube(launch)]
pub fn median_filter_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
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

/// [`fn@med3`] lane by lane.
#[cube]
fn med3_vector<F: Float, N: Size>(a: Vector<F, N>, b: Vector<F, N>, c: Vector<F, N>) -> Vector<F, N> {
    a.min(b).max(a.max(b).min(c))
}

/// [`fn@med9`] lane by lane (the same min/max network).
#[cube]
fn med9_vector<F: Float, N: Size>(w: &Array<Vector<F, N>>) -> Vector<F, N> {
    let max_min = w[0].min(w[1].min(w[2])).max(w[3].min(w[4].min(w[5])).max(w[6].min(w[7].min(w[8]))));
    let min_max = w[0].max(w[1].max(w[2])).min(w[3].max(w[4].max(w[5])).min(w[6].max(w[7].max(w[8]))));
    let mid_mid = med3_vector::<F, N>(med3_vector::<F, N>(w[0], w[1], w[2]), med3_vector::<F, N>(w[3], w[4], w[5]), med3_vector::<F, N>(w[6], w[7], w[8]));
    med3_vector::<F, N>(max_min, mid_mid, min_max)
}

/// [`fn@median_by_rank`] lane by lane.
#[cube]
fn median_by_rank_vector<F: Float, N: Size>(window: &Array<Vector<F, N>>, #[comptime] width: u32) -> Vector<F, N> {
    let half = Vector::<u32, N>::new(comptime!(width / 2));
    let one = Vector::<u32, N>::new(1u32);
    let zero = Vector::<u32, N>::new(0u32);
    let mut median = window[0usize];
    // Unrolled for short windows only: a wide window unrolled over every lane is a very large
    // kernel to compile
    #[unroll(width <= 9)]
    for i in 0..width {
        let x = window[i as usize];
        let mut less = zero;
        let mut less_equal = zero;
        #[unroll(width <= 9)]
        for j in 0..width {
            let y = window[j as usize];
            less += select_many(y.less_than(&x), one, zero);
            less_equal += select_many(y.less_equal(&x), one, zero);
        }
        median = select_many(less.less_equal(&half).vec_and(less_equal.greater_than(&half)), x, median);
    }
    median
}

/// [`fn@median_filter_kernel`] with `N` consecutive outputs per unit (`num_samples` a multiple of
/// `N`). The unit reads its `width + N − 1` samples once and assembles window vector `k` (lane `l`
/// = sample `i + l − radius + k`) from them; the median network or rank count then runs lane by
/// lane with the same comparisons as the scalar kernel, so the output is bit-identical.
#[cube(launch)]
pub fn median_filter_vector_kernel<F: Float, N: Size>(
    input: &[F],
    output: &mut [Vector<F, N>],
    num_samples: u32,
    #[comptime] width: u32,
    #[comptime] edge: u32,
) {
    let v = element_position();
    if v < output.len() {
        let mut window = Array::<Vector<F, N>>::new(comptime!(width as usize));
        let n = window[0usize].vector_size();
        let span = comptime!(width as usize + n - 1);
        let radius = comptime!(width / 2);
        let per_row = num_samples / n as u32;
        let i = (v as u32 % per_row) * n as u32;
        let base = ((v as u32 / per_row) * num_samples) as usize;
        let mut raw = Array::<F>::new(span);
        if i >= radius && i + n as u32 - 1u32 + radius < num_samples {
            let first = base + (i - radius) as usize;
            #[unroll(width <= 9)]
            for k in 0..span {
                raw[k] = input[first + k];
            }
        } else {
            // Logical sample i − radius + k is extended position i + k
            #[unroll(width <= 9)]
            for k in 0..span {
                raw[k] = read_extended::<F>(input, base, num_samples, radius, i + k as u32, edge);
            }
        }
        #[unroll(width <= 9)]
        for k in 0..width {
            let mut w = Vector::<F, N>::empty();
            #[unroll]
            for l in 0..n {
                w.insert(l, raw[k as usize + l]);
            }
            window[k as usize] = w;
        }
        let median = if comptime!(width == 9) { med9_vector::<F, N>(&window) } else { median_by_rank_vector::<F, N>(&window, width) };
        output[v] = median;
    }
}
