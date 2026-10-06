//! Point-cloud kernels of HDBSCAN and k-means. Points are stored **feature-major**
//! (`x[f · n + i]`, `[d, n]`): units walking neighbouring points read neighbouring addresses, and
//! the dsp-base row reductions (per-feature moments, second moment) apply unchanged.

use cubecl::prelude::*;
use dsp_core::compute::row_position;

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

/// Copies the features of point `i` (feature-major `[d, n]`) into `own` (registers).
#[cube]
fn load_own<F: Float>(x: &Array<F>, own: &mut Array<F>, i: u32, n: u32, #[comptime] d: u32) {
    let mut f = 0u32;
    while f < d {
        own[f as usize] = x[(f * n + i) as usize];
        f += 1u32;
    }
}

/// Unit `UNIT_POS_X` of the cube copies point `t0 + UNIT_POS_X` (if before `j1`) into its row of
/// the shared tile (`[tile, d]`).
#[cube]
fn load_tile<F: Float>(x: &Array<F>, tile: &mut SharedMemory<F>, t0: u32, j1: u32, n: u32, #[comptime] d: u32) {
    let j = t0 + UNIT_POS_X;
    if j < j1 {
        let mut f = 0u32;
        while f < d {
            tile[(UNIT_POS_X * d + f) as usize] = x[(f * n + j) as usize];
            f += 1u32;
        }
    }
}

/// `‖own − tile[u]‖²`.
#[cube]
fn tile_sq_dist<F: Float>(own: &Array<F>, tile: &SharedMemory<F>, u: u32, #[comptime] d: u32) -> F {
    let mut acc = F::new(0.0f32);
    let mut f = 0u32;
    while f < d {
        let diff = own[f as usize] - tile[(u * d + f) as usize];
        acc += diff * diff;
        f += 1u32;
    }
    acc
}

/// HDBSCAN core distances over points `j0..j1` (one launch of several): `best[i, ·]` (`[n, k]`)
/// keeps the `k` smallest squared distances from point `i` so far, sorted, and `core[i]` the
/// `k`-th (the point itself counts as the first; `min_samples = k`). Points `j` are shared by
/// the cube in tiles of `units` (its size) through shared memory. Launch with
/// [`dsp_core::compute::LaunchGeometry::tiles`]; `best` starts at `F::max_value()`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn core_distance_tile_kernel<F: Float>(
    x: &Array<F>,
    best: &mut Array<F>,
    core: &mut Array<F>,
    n: u32,
    j0: u32,
    j1: u32,
    #[comptime] d: u32,
    #[comptime] k: u32,
    #[comptime] units: u32,
) {
    let i = row_position() * units + UNIT_POS_X;
    let active = i < n;
    let last = comptime!(k - 1);
    let mut kbest = Array::<F>::new(comptime!(k as usize));
    let mut own = Array::<F>::new(comptime!(d as usize));
    if active {
        load_own::<F>(x, &mut own, i, n, d);
        #[unroll]
        for s in 0..k {
            kbest[s as usize] = best[(i * k + s) as usize];
        }
    }
    let mut tile = SharedMemory::<F>::new(comptime!((units * d) as usize));
    let mut t0 = j0;
    while t0 < j1 {
        load_tile::<F>(x, &mut tile, t0, j1, n, d);
        sync_cube();
        if active {
            let count = u32::min(units, j1 - t0);
            let mut u = 0u32;
            while u < count {
                let dist = tile_sq_dist::<F>(&own, &tile, u, d);
                if dist < kbest[last as usize] {
                    // Insertion: shift the larger ones up one slot
                    let pos = RuntimeCell::<u32>::new(last);
                    let mut moving = true;
                    while moving {
                        let s = pos.read();
                        if s > 0u32 {
                            if kbest[(s - 1u32) as usize] > dist {
                                kbest[s as usize] = kbest[(s - 1u32) as usize];
                                pos.store(s - 1u32);
                            } else {
                                moving = false;
                            }
                        } else {
                            moving = false;
                        }
                    }
                    kbest[pos.read() as usize] = dist;
                }
                u += 1u32;
            }
        }
        sync_cube();
        t0 += units;
    }
    if active {
        #[unroll]
        for s in 0..k {
            best[(i * k + s) as usize] = kbest[s as usize];
        }
        core[i as usize] = kbest[last as usize];
    }
}

/// Borůvka step of HDBSCAN's minimum spanning tree over the mutual-reachability distance
/// `max(core_i, core_j, ‖x_i − x_j‖²)` (all squared), for points `j0..j1` (one launch of
/// several): `best_w[i]`, `best_j[i]` keep point `i`'s cheapest edge to another component so far
/// (the smallest `j` on ties; `best_j[i] = n` when there is none). Launches run `j` in increasing
/// order; the first resets the state (`first = 1`). Points `j`, their core distances and
/// components are shared by the cube in tiles. Launch with
/// [`dsp_core::compute::LaunchGeometry::tiles`].
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cheapest_edge_tile_kernel<F: Float>(
    x: &Array<F>,
    core: &Array<F>,
    component: &Array<u32>,
    best_w: &mut Array<F>,
    best_j: &mut Array<u32>,
    n: u32,
    j0: u32,
    j1: u32,
    first: u32,
    #[comptime] d: u32,
    #[comptime] units: u32,
) {
    let i = row_position() * units + UNIT_POS_X;
    let active = i < n;
    let mut own = Array::<F>::new(comptime!(d as usize));
    let mut ci = 0u32;
    let mut core_i = F::new(0.0f32);
    let mut bw = F::max_value();
    let mut bj = n;
    if active {
        load_own::<F>(x, &mut own, i, n, d);
        ci = component[i as usize];
        core_i = core[i as usize];
        if first == 0u32 {
            bw = best_w[i as usize];
            bj = best_j[i as usize];
        }
    }
    let mut tile = SharedMemory::<F>::new(comptime!((units * d) as usize));
    let mut tile_core = SharedMemory::<F>::new(comptime!(units as usize));
    let mut tile_comp = SharedMemory::<u32>::new(comptime!(units as usize));
    let mut t0 = j0;
    while t0 < j1 {
        load_tile::<F>(x, &mut tile, t0, j1, n, d);
        let j = t0 + UNIT_POS_X;
        if j < j1 {
            tile_core[UNIT_POS_X as usize] = core[j as usize];
            tile_comp[UNIT_POS_X as usize] = component[j as usize];
        }
        sync_cube();
        if active {
            let count = u32::min(units, j1 - t0);
            let mut u = 0u32;
            while u < count {
                if tile_comp[u as usize] != ci {
                    let w = F::max(F::max(core_i, tile_core[u as usize]), tile_sq_dist::<F>(&own, &tile, u, d));
                    if w < bw {
                        bw = w;
                        bj = t0 + u;
                    }
                }
                u += 1u32;
            }
        }
        sync_cube();
        t0 += units;
    }
    if active {
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
