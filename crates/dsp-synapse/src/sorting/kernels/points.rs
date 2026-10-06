//! Point-cloud kernels of HDBSCAN and k-means. Points are stored **feature-major**
//! (`x[f · n + i]`, `[d, n]`): units walking neighbouring points read neighbouring addresses, and
//! the dsp-base row reductions (per-feature moments, second moment) apply unchanged.

use cubecl::prelude::*;

/// `‖x_a − x_b‖²` of points `a` and `b` of a feature-major `[d, n]` buffer.
#[cube]
pub fn point_sq_dist<F: Float>(x: &Array<F>, a: u32, b: u32, n: u32, d: u32) -> F {
    let mut acc = F::new(0.0f32);
    let mut f = 0u32;
    while f < d {
        let diff = x[(f * n + a) as usize] - x[(f * n + b) as usize];
        acc += diff * diff;
        f += 1u32;
    }
    acc
}

/// `core[i]` = squared distance from point `i` to its `k`-th nearest point, itself counted first
/// (HDBSCAN core distance with `min_samples = k`). The `k` smallest are kept sorted in registers.
/// One unit per point.
#[cube(launch)]
pub fn core_distance_kernel<F: Float>(x: &Array<F>, core: &mut Array<F>, n: u32, d: u32, #[comptime] k: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let last = comptime!(k - 1);
        let mut best = Array::<F>::new(comptime!(k as usize));
        #[unroll]
        for s in 0..k {
            best[s as usize] = F::max_value();
        }
        let mut j = 0u32;
        while j < n {
            let dist = point_sq_dist::<F>(x, i, j, n, d);
            if dist < best[last as usize] {
                // Insertion: shift the larger ones up one slot
                let pos = RuntimeCell::<u32>::new(last);
                let mut moving = true;
                while moving {
                    let s = pos.read();
                    if s > 0u32 {
                        if best[(s - 1u32) as usize] > dist {
                            best[s as usize] = best[(s - 1u32) as usize];
                            pos.store(s - 1u32);
                        } else {
                            moving = false;
                        }
                    } else {
                        moving = false;
                    }
                }
                best[pos.read() as usize] = dist;
            }
            j += 1u32;
        }
        core[i as usize] = best[last as usize];
    }
}

/// Borůvka step of HDBSCAN's minimum spanning tree over the mutual-reachability distance
/// `max(core_i, core_j, ‖x_i − x_j‖²)` (all squared): the cheapest edge from point `i` to a
/// point of another component, the smallest `j` on ties (`best_j[i] = n` when there is none).
/// One unit per point.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cheapest_edge_kernel<F: Float>(
    x: &Array<F>,
    core: &Array<F>,
    component: &Array<u32>,
    best_w: &mut Array<F>,
    best_j: &mut Array<u32>,
    n: u32,
    d: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let ci = component[i as usize];
        let core_i = core[i as usize];
        let mut bw = F::max_value();
        let mut bj = n;
        let mut j = 0u32;
        while j < n {
            if component[j as usize] != ci {
                let w = F::max(F::max(core_i, core[j as usize]), point_sq_dist::<F>(x, i, j, n, d));
                if w < bw {
                    bw = w;
                    bj = j;
                }
            }
            j += 1u32;
        }
        best_w[i as usize] = bw;
        best_j[i as usize] = bj;
    }
}

/// k-means assignment: `label[i]` = nearest of the `k` centres (`[k, d]` row-major; the first on
/// ties) and `dist[i]` its squared distance. One unit per point.
#[cube(launch)]
pub fn nearest_centre_kernel<F: Float>(
    x: &Array<F>,
    centres: &Array<F>,
    label: &mut Array<u32>,
    dist: &mut Array<F>,
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
    x: &Array<F>,
    label: &Array<u32>,
    sums: &mut Array<F>,
    counts: &mut Array<u32>,
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
    x: &Array<F>,
    closest: &Array<F>,
    updated: &mut Array<F>,
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
pub fn block_sums_kernel<F: Float>(values: &Array<F>, sums: &mut Array<F>, n: u32, block: u32, blocks: u32) {
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
pub fn gather_points_kernel<F: Float>(x: &Array<F>, index: &Array<u32>, out: &mut Array<F>, n: u32, m: u32, d: u32) {
    let unit = ABSOLUTE_POS as u32;
    if unit < m * d {
        let f = unit / m;
        let j = unit - f * m;
        out[unit as usize] = x[(f * n + index[j as usize]) as usize];
    }
}
