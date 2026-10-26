//! Point-cloud kernels of HDBSCAN and k-means. Points are stored **feature-major**
//! (`x[f · n + i]`, `[d, n]`): units walking neighbouring points read neighbouring addresses, and
//! the dsp-base row reductions (per-feature moments, second moment) apply unchanged.

use cubecl::prelude::*;

/// `‖x_a − x_b‖²` of points `a` and `b` of a feature-major `[d, n]` buffer.
#[cube]
pub fn point_sq_dist<F: Float>(x: &[F], a: u32, b: u32, n: u32, d: u32) -> F {
    let mut acc = F::new(0.0f32);
    let mut f = 0u32;
    while f < d {
        let diff = x[(f * n + a) as usize] - x[(f * n + b) as usize];
        acc += diff * diff;
        f += 1u32;
    }
    acc
}

/// Copies the features of point `i` (feature-major `[d, n]`) into `own`. The loop is unrolled
/// (`d` is comptime), so every index is a constant and `own` can stay in registers.
#[cube]
pub(crate) fn load_own<F: Float>(x: &[F], own: &mut Array<F>, i: u32, n: u32, #[comptime] d: u32) {
    #[unroll]
    for f in 0..d {
        own[f as usize] = x[(f * n + i) as usize];
    }
}

/// Unit `UNIT_POS_X` of the cube copies point `t0 + UNIT_POS_X` (if before `j1`) into its row of
/// the shared tile (`[tile, d]`).
#[cube]
pub(crate) fn load_tile<F: Float>(x: &[F], tile: &mut Shared<[F]>, t0: u32, j1: u32, n: u32, #[comptime] d: u32) {
    let j = t0 + UNIT_POS_X;
    if j < j1 {
        #[unroll]
        for f in 0..d {
            tile[(UNIT_POS_X * d + f) as usize] = x[(f * n + j) as usize];
        }
    }
}

/// `‖own − tile[u]‖²`, summed over the features in order (unrolled; same order as
/// [`point_sq_dist`]).
#[cube]
pub(crate) fn tile_sq_dist<F: Float>(own: &Array<F>, tile: &Shared<[F]>, u: u32, #[comptime] d: u32) -> F {
    let mut acc = F::new(0.0f32);
    #[unroll]
    for f in 0..d {
        let diff = own[f as usize] - tile[(u * d + f) as usize];
        acc += diff * diff;
    }
    acc
}

/// k-means assignment: `label[i]` = nearest of the `k` centres (`[k, d]` row-major; the first on
/// ties) and `dist[i]` its squared distance. One unit per point.
#[cube(launch)]
pub fn nearest_centre_kernel<F: Float>(
    x: &[F],
    centres: &[F],
    label: &mut [u32],
    dist: &mut [F],
    n: u32,
    d: u32,
    k: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let mut best = F::max_value();
        let mut best_c = 0u32;
        let mut c = 0u32;
        while c < k {
            let mut acc = F::new(0.0f32);
            let mut f = 0u32;
            while f < d {
                let diff = x[(f * n + i) as usize] - centres[(c * d + f) as usize];
                acc += diff * diff;
                f += 1u32;
            }
            if acc < best {
                best = acc;
                best_c = c;
            }
            c += 1u32;
        }
        label[i as usize] = best_c;
        dist[i as usize] = best;
    }
}

/// k-means update sums over a split of the points: `sums[s, c, f] = Σ_{i ∈ split s, label[i] = c}
/// x[f, i]` and `counts[s, c]` (written by `f = 0`). Splits keep each `f32` sum short; the host
/// adds them in `f64`. One unit per `(s, c, f)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cluster_sums_kernel<F: Float>(
    x: &[F],
    label: &[u32],
    sums: &mut [F],
    counts: &mut [u32],
    n: u32,
    d: u32,
    k: u32,
    split_len: u32,
    splits: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < splits * k * d {
        let s = unit / (k * d);
        let rest = unit - s * k * d;
        let c = rest / d;
        let f = rest - c * d;
        let end = u32::min((s + 1u32) * split_len, n);
        let mut acc = F::new(0.0f32);
        let mut count = 0u32;
        let mut i = s * split_len;
        while i < end {
            if label[i as usize] == c {
                acc += x[(f * n + i) as usize];
                count += 1u32;
            }
            i += 1u32;
        }
        sums[unit as usize] = acc;
        if f == 0u32 {
            counts[(s * k + c) as usize] = count;
        }
    }
}

/// k-means++ trial: `updated[i] = min(closest[i], ‖x_i − x_candidate‖²)`. One unit per point.
#[cube(launch)]
pub fn closest_update_kernel<F: Float>(
    x: &[F],
    closest: &[F],
    updated: &mut [F],
    candidate: u32,
    n: u32,
    d: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        updated[i as usize] = F::min(closest[i as usize], point_sq_dist::<F>(x, i, candidate, n, d));
    }
}

/// `sums[b] = Σ values[b · block .. min((b + 1) · block, n)]`. One unit per block.
#[cube(launch)]
pub fn block_sums_kernel<F: Float>(values: &[F], sums: &mut [F], n: u32, block: u32, blocks: u32) {
    let b = ABSOLUTE_POS as u32;
    if b < blocks {
        let end = u32::min((b + 1u32) * block, n);
        let mut acc = F::new(0.0f32);
        let mut i = b * block;
        while i < end {
            acc += values[i as usize];
            i += 1u32;
        }
        sums[b as usize] = acc;
    }
}

/// `out[f, j] = x[f, index[j]]`: the points `index` of a feature-major `[d, n]` buffer as a
/// feature-major `[d, m]` buffer. One unit per `(f, j)`.
#[cube(launch)]
pub fn gather_points_kernel<F: Float>(x: &[F], index: &[u32], out: &mut [F], n: u32, m: u32, d: u32) {
    let unit = ABSOLUTE_POS as u32;
    if unit < m * d {
        let f = unit / m;
        let j = unit - f * m;
        out[unit as usize] = x[(f * n + index[j as usize]) as usize];
    }
}
