//! HDBSCAN kernels (see `sorting/hdbscan.rs` for the algorithm). Points are feature-major
//! (`x[f · n + i]`); squared distances throughout.
//!
//! Every loop here has a comptime or counter bound: a loop driven by a flag a branch clears is
//! the one construct a compiler change can turn into a loop that never exits (a device hang), so
//! none is used.

use cubecl::prelude::*;
use dsp_core::compute::row_position;

use super::points::{load_own, load_tile, tile_sq_dist};

/// Nearest neighbours over points `j0..j1` (one launch of several): `best_d[i, ·]` / `best_j[i, ·]`
/// (`[n, list]`) keep the `list` smallest squared distances from point `i` so far and their
/// points, sorted by distance (on equal distances the earlier `j` first), and `core[i]` the entry
/// at `rank` (the point itself counts as the first: `rank = min_samples − 1`). Keeping one entry
/// more than `min_samples` tells the spanning-tree step whether the list holds every point within
/// the core distance. Points `j` are shared by the cube in tiles of `units` (its size) through
/// shared memory. Launch with [`dsp_core::compute::LaunchGeometry::tiles`]; `best_d` starts at
/// `F::max_value()`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn knn_tile_kernel<F: Float>(
    x: &[F],
    best_d: &mut [F],
    best_j: &mut [u32],
    core: &mut [F],
    n: u32,
    j0: u32,
    j1: u32,
    #[comptime] d: u32,
    #[comptime] list: u32,
    #[comptime] rank: u32,
    #[comptime] units: u32,
) {
    let i = row_position() * units + UNIT_POS_X;
    let active = i < n;
    let last = comptime!(list - 1);
    let mut kd = Array::<F>::new(comptime!(list as usize));
    let mut kj = Array::<u32>::new(comptime!(list as usize));
    let mut own = Array::<F>::new(comptime!(d as usize));
    if active {
        load_own::<F>(x, &mut own, i, n, d);
        #[unroll]
        for s in 0..list {
            kd[s as usize] = best_d[(i * list + s) as usize];
            kj[s as usize] = best_j[(i * list + s) as usize];
        }
    }
    let mut tile = Shared::<[F]>::new_slice(comptime!((units * d) as usize));
    let mut t0 = j0;
    while t0 < j1 {
        load_tile::<F>(x, &mut tile, t0, j1, n, d);
        sync_cube();
        if active {
            let count = u32::min(units, j1 - t0);
            let mut u = 0u32;
            while u < count {
                let dist = tile_sq_dist::<F>(&own, &tile, u, d);
                if dist < kd[last as usize] {
                    // Replace the largest, then move it down past every strictly larger entry
                    // (a fixed number of compare-and-swaps: equal distances keep their order)
                    kd[last as usize] = dist;
                    kj[last as usize] = t0 + u;
                    #[unroll]
                    for step in 0..last {
                        let s = comptime!(last - step);
                        if kd[s as usize] < kd[(s - 1) as usize] {
                            let (dd, jj) = (kd[s as usize], kj[s as usize]);
                            kd[s as usize] = kd[(s - 1) as usize];
                            kj[s as usize] = kj[(s - 1) as usize];
                            kd[(s - 1) as usize] = dd;
                            kj[(s - 1) as usize] = jj;
                        }
                    }
                }
                u += 1u32;
            }
        }
        sync_cube();
        t0 += units;
    }
    if active {
        #[unroll]
        for s in 0..list {
            best_d[(i * list + s) as usize] = kd[s as usize];
            best_j[(i * list + s) as usize] = kj[s as usize];
        }
        core[i as usize] = kd[rank as usize];
    }
}

/// Each point's cheapest edge to another component **among its nearest neighbours**, one unit per
/// point: for the first `within` entries of its list (all within its core distance, so the
/// mutual-reachability weight is `max(core_i, core_j)`), the smallest weight, the smallest `j` on
/// ties (`cand_j = n` when none is in another component). `resolved[i] = 1` when that edge is
/// provably the point's cheapest overall: its weight equals `core_i` (no edge of `i` is lighter)
/// and the entry after the first `within` lies beyond `core_i` (every point within the core
/// distance is in the list, so no smaller `j` of equal weight is missing).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn knn_candidate_kernel<F: Float>(
    best_d: &[F],
    best_j: &[u32],
    core: &[F],
    component: &[u32],
    cand_w: &mut [F],
    cand_j: &mut [u32],
    resolved: &mut [u32],
    n: u32,
    #[comptime] list: u32,
    #[comptime] within: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let (ci, core_i) = (component[i as usize], core[i as usize]);
        let mut bw = F::max_value();
        let mut bj = n;
        #[unroll]
        for s in 0..within {
            let j = best_j[(i * list + s) as usize];
            let dj = best_d[(i * list + s) as usize];
            if j < n && j != i && dj <= core_i {
                if component[j as usize] != ci {
                    let w = F::max(core_i, core[j as usize]);
                    if w < bw || (w == bw && j < bj) {
                        bw = w;
                        bj = j;
                    }
                }
            }
        }
        cand_w[i as usize] = bw;
        cand_j[i as usize] = bj;
        let mut done = 0u32;
        if bj < n && bw == core_i && best_d[(i * list + within) as usize] > core_i {
            done = 1u32;
        }
        resolved[i as usize] = done;
    }
}

/// Borůvka step over the mutual-reachability distance `max(core_i, core_j, ‖x_i − x_j‖²)` for the
/// `n_active` points listed in `active`, against points `j0..j1` (one launch of several):
/// `best_w[a]`, `best_j[a]` keep active point `a`'s cheapest edge to another component so far (the
/// smallest `j` on ties; `best_j[a] = n` when none). Launches run `j` in increasing order; the
/// first resets the state (`first = 1`). Points `j`, their core distances and components are
/// shared by the cube in tiles. Launch with [`dsp_core::compute::LaunchGeometry::tiles`] over
/// `n_active` points.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cheapest_edge_tile_kernel<F: Float>(
    x: &[F],
    core: &[F],
    component: &[u32],
    active_points: &[u32],
    best_w: &mut [F],
    best_j: &mut [u32],
    n: u32,
    n_active: u32,
    j0: u32,
    j1: u32,
    first: u32,
    #[comptime] d: u32,
    #[comptime] units: u32,
) {
    let a = row_position() * units + UNIT_POS_X;
    let active = a < n_active;
    let mut own = Array::<F>::new(comptime!(d as usize));
    let mut ci = 0u32;
    let mut core_i = F::new(0.0f32);
    let mut bw = F::max_value();
    let mut bj = n;
    if active {
        let i = active_points[a as usize];
        load_own::<F>(x, &mut own, i, n, d);
        ci = component[i as usize];
        core_i = core[i as usize];
        if first == 0u32 {
            bw = best_w[a as usize];
            bj = best_j[a as usize];
        }
    }
    let mut tile = Shared::<[F]>::new_slice(comptime!((units * d) as usize));
    let mut tile_core = Shared::<[F]>::new_slice(comptime!(units as usize));
    let mut tile_comp = Shared::<[u32]>::new_slice(comptime!(units as usize));
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
        best_w[a as usize] = bw;
        best_j[a as usize] = bj;
    }
}
