//! Bipartite-graph clustering kernels (see `sorting/bipartite.rs` for the algorithm). Points are
//! feature-major (`x[f · n + i]`); the graph is `nb[t · k + s]`: the `k` right nodes (subset
//! points) linked to left node `t`. Labels are `u32`; [`NO_LABEL`] marks a node without one.
//!
//! Counters are global `Atomic<u32>` reached by reference (`Atomic::fetch_add(&c[i], 1)`).

use cubecl::prelude::*;
use dsp_base::core::DspFloat;

/// Label of a node outside every cluster (a right node no left node of a kept cluster reaches).
pub const NO_LABEL: u32 = u32::MAX;

/// `out[i] = Σ_f x[f, i]²`. One unit per point.
#[cube(launch)]
pub fn sq_norms_kernel<F: Float>(x: &[F], out: &mut [F], n: u32, d: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let mut acc = F::new(0.0f32);
        let mut f = 0u32;
        while f < d {
            let v = x[(f * n + i) as usize];
            acc += v * v;
            f += 1u32;
        }
        out[i as usize] = acc;
    }
}

/// The `k` nearest subset points of rows `row0 .. row0 + rows` of the left points, from their
/// inner products `dots[r, j]` (`[rows, m]`): squared distance `‖x‖² + ‖y_j‖² − 2·dot`, the subset
/// point that **is** the row's point (`subset[j] == row`) skipped. Kept sorted by distance (on
/// equal distances the smaller `j` first); `nb[row · k + s]`. One unit per row.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn select_neighbours_kernel<F: Float>(
    dots: &[F],
    left_norms: &[F],
    right_norms: &[F],
    subset: &[u32],
    nb: &mut [u32],
    row0: u32,
    rows: u32,
    m: u32,
    #[comptime] k: u32,
) {
    let r = ABSOLUTE_POS as u32;
    if r < rows {
        let row = row0 + r;
        let last = comptime!(k - 1);
        let mut kd = Array::<F>::new(comptime!(k as usize));
        let mut kj = Array::<u32>::new(comptime!(k as usize));
        #[unroll]
        for s in 0..k {
            kd[s as usize] = F::max_value();
            kj[s as usize] = 0u32;
        }
        let own = left_norms[row as usize];
        let mut j = 0u32;
        while j < m {
            if subset[j as usize] != row {
                let dist = own + right_norms[j as usize] - F::new(2.0f32) * dots[(r * m + j) as usize];
                if dist < kd[last as usize] {
                    // Replace the largest, then move it down past every strictly larger entry
                    kd[last as usize] = dist;
                    kj[last as usize] = j;
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
            }
            j += 1u32;
        }
        #[unroll]
        for s in 0..k {
            nb[(row * k + s) as usize] = kj[s as usize];
        }
    }
}

/// `counts[v] += 1` for every value `v = values[i] < bins` (`counts` zeroed by the caller).
#[cube(launch)]
pub fn count_values_kernel(values: &[u32], counts: &mut [Atomic<u32>], n: u32, bins: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let v = values[i as usize];
        if v < bins {
            Atomic::fetch_add(&counts[v as usize], 1u32);
        }
    }
}

/// Every edge `(t, r)` adds 1 to `counts[r · c + left[t]]` (`[m, c]`, zeroed): how many of each
/// right node's left neighbours sit in each cluster. One unit per edge.
#[cube(launch)]
pub fn right_counts_kernel(nb: &[u32], left: &[u32], counts: &mut [Atomic<u32>], edges: u32, k: u32, c: u32) {
    let e = ABSOLUTE_POS as u32;
    if e < edges {
        let t = e / k;
        let label = left[t as usize];
        if label < c {
            Atomic::fetch_add(&counts[(nb[e as usize] * c + label) as usize], 1u32);
        }
    }
}

/// Right step: each right node `r` joins the cluster `c` among its neighbours' maximizing
/// `counts[r, c] − γ · deg[r] · k_left[c] / 2m` (the paper's bipartite modularity gain), the
/// smallest `c` on ties; a node without edges keeps its label. One unit per right node.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn right_assign_kernel<F: DspFloat>(
    counts: &[u32],
    degree: &[u32],
    k_left: &[u32],
    right: &mut [u32],
    m: u32,
    c: u32,
    penalty: F,
) {
    let r = ABSOLUTE_POS as u32;
    if r < m {
        let deg = F::cast_from(degree[r as usize]);
        let mut best = F::min_value();
        let mut best_c = right[r as usize];
        let mut found = false;
        let mut j = 0u32;
        while j < c {
            let n_rc = counts[(r * c + j) as usize];
            if n_rc > 0u32 {
                let gain = F::cast_from(n_rc) - penalty * deg * F::cast_from(k_left[j as usize]);
                if !found || gain > best {
                    best = gain;
                    best_c = j;
                    found = true;
                }
            }
            j += 1u32;
        }
        right[r as usize] = best_c;
    }
}

/// `sums[right[r]] += degree[r]` for every labelled right node (`sums` zeroed).
#[cube(launch)]
pub fn degree_sums_kernel(labels: &[u32], degree: &[u32], sums: &mut [Atomic<u32>], m: u32, c: u32) {
    let r = ABSOLUTE_POS as u32;
    if r < m {
        let label = labels[r as usize];
        if label < c {
            Atomic::fetch_add(&sums[label as usize], degree[r as usize]);
        }
    }
}

/// Left step: each left node `t` joins the cluster among its `k` right neighbours' labels
/// maximizing `n_tc − γ · k · k_right[c] / 2m` (every left node has degree `k`), the smallest
/// label on ties; `changed[0]` counts the nodes that moved. One unit per left node.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn left_assign_kernel<F: DspFloat>(
    nb: &[u32],
    right: &[u32],
    k_right: &[u32],
    left: &mut [u32],
    changed: &mut [Atomic<u32>],
    n: u32,
    c: u32,
    penalty: F,
    #[comptime] k: u32,
) {
    let t = ABSOLUTE_POS as u32;
    if t < n {
        let mut labels = Array::<u32>::new(comptime!(k as usize));
        #[unroll]
        for s in 0..k {
            labels[s as usize] = right[nb[(t * k + s) as usize] as usize];
        }
        let deg = F::cast_from(k);
        let mut best = F::min_value();
        let mut best_c = left[t as usize];
        let mut found = false;
        #[unroll]
        for s in 0..k {
            let label = labels[s as usize];
            if label < c {
                let mut n_tc = 0u32;
                #[unroll]
                for q in 0..k {
                    if labels[q as usize] == label {
                        n_tc += 1u32;
                    }
                }
                let gain = F::cast_from(n_tc) - penalty * deg * F::cast_from(k_right[label as usize]);
                if !found || gain > best || (gain == best && label < best_c) {
                    best = gain;
                    best_c = label;
                    found = true;
                }
            }
        }
        if best_c != left[t as usize] {
            left[t as usize] = best_c;
            Atomic::fetch_add(&changed[0], 1u32);
        }
    }
}

/// `labels[i] = map[labels[i]]` (labels past the map become [`NO_LABEL`]).
#[cube(launch)]
pub fn relabel_kernel(labels: &mut [u32], map: &[u32], n: u32, c: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let label = labels[i as usize];
        if label < c {
            labels[i as usize] = map[label as usize];
        } else {
            labels[i as usize] = NO_LABEL;
        }
    }
}

/// Edge counts between clusters: every edge `(t, r)` adds 1 to `edges[left[t] · c + right[r]]`
/// (`[c, c]`, zeroed). One unit per edge.
#[cube(launch)]
pub fn cluster_edges_kernel(nb: &[u32], left: &[u32], right: &[u32], edges: &mut [Atomic<u32>], n_edges: u32, k: u32, c: u32) {
    let e = ABSOLUTE_POS as u32;
    if e < n_edges {
        let a = left[(e / k) as usize];
        let b = right[nb[e as usize] as usize];
        if a < c && b < c {
            Atomic::fetch_add(&edges[(a * c + b) as usize], 1u32);
        }
    }
}

/// k-means++ trials scored at once: `partial[t · blocks + b] = Σ_{i ∈ block b} min(closest[i],
/// ‖x_i − x_{cand[t]}‖²)`, the potential if trial `t` became a seed, per block of `block` points
/// (the host adds the blocks in `f64`). One unit per `(t, b)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn trial_potentials_kernel<F: Float>(
    x: &[F],
    closest: &[F],
    cand: &[u32],
    partial: &mut [F],
    n: u32,
    d: u32,
    trials: u32,
    block: u32,
    blocks: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < trials * blocks {
        let t = unit / blocks;
        let b = unit - t * blocks;
        let c = cand[t as usize];
        let end = u32::min((b + 1u32) * block, n);
        let mut acc = F::new(0.0f32);
        let mut i = b * block;
        while i < end {
            let mut dist = F::new(0.0f32);
            let mut f = 0u32;
            while f < d {
                let diff = x[(f * n + i) as usize] - x[(f * n + c) as usize];
                dist += diff * diff;
                f += 1u32;
            }
            acc += F::min(dist, closest[i as usize]);
            i += 1u32;
        }
        partial[unit as usize] = acc;
    }
}

/// k-means++ draws of seed `s`, one unit per trial `t`: the first point whose running sum of
/// `closest` passes `uniforms[s · trials + t] · total` (walking the `blocks` sums `sums`, then the
/// block), the last point on round-off. `cand[t]` receives it.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn draw_candidates_kernel<F: Float>(
    closest: &[F],
    sums: &[F],
    uniforms: &[F],
    cand: &mut [u32],
    n: u32,
    block: u32,
    blocks: u32,
    trials: u32,
    s: u32,
) {
    let t = ABSOLUTE_POS as u32;
    if t < trials {
        let mut total = F::new(0.0f32);
        let mut b = 0u32;
        while b < blocks {
            total += sums[b as usize];
            b += 1u32;
        }
        let mut r = uniforms[(s * trials + t) as usize] * total;
        // Block holding the draw (the last one on round-off)
        let mut chosen = blocks - 1u32;
        let mut found = false;
        b = 0u32;
        while b < blocks {
            if !found {
                let sb = sums[b as usize];
                if r < sb {
                    chosen = b;
                    found = true;
                } else {
                    r -= sb;
                }
            }
            b += 1u32;
        }
        let start = chosen * block;
        let end = u32::min(start + block, n);
        let mut pick = end - 1u32;
        let mut hit = false;
        let mut i = start;
        while i < end {
            if !hit {
                let w = closest[i as usize];
                if r < w {
                    pick = i;
                    hit = true;
                } else {
                    r -= w;
                }
            }
            i += 1u32;
        }
        cand[t as usize] = pick;
    }
}

/// The trial with the lowest potential (the first on ties) becomes seed `s`: `seeds[s]` and
/// `chosen[0]` receive its point. One unit.
#[cube(launch)]
pub fn pick_seed_kernel<F: Float>(partial: &[F], cand: &[u32], seeds: &mut [u32], chosen: &mut [u32], trials: u32, blocks: u32, s: u32) {
    if ABSOLUTE_POS == 0 {
        let mut best = F::max_value();
        let mut best_t = 0u32;
        let mut t = 0u32;
        while t < trials {
            let mut pot = F::new(0.0f32);
            let mut b = 0u32;
            while b < blocks {
                pot += partial[(t * blocks + b) as usize];
                b += 1u32;
            }
            if pot < best {
                best = pot;
                best_t = t;
            }
            t += 1u32;
        }
        seeds[s as usize] = cand[best_t as usize];
        chosen[0] = cand[best_t as usize];
    }
}

/// `closest[i] = min(closest[i], ‖x_i − x_{chosen[0]}‖²)`, in place. One unit per point.
#[cube(launch)]
pub fn closest_update_chosen_kernel<F: Float>(x: &[F], closest: &mut [F], chosen: &[u32], n: u32, d: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let c = chosen[0];
        let mut dist = F::new(0.0f32);
        let mut f = 0u32;
        while f < d {
            let diff = x[(f * n + i) as usize] - x[(f * n + c) as usize];
            dist += diff * diff;
            f += 1u32;
        }
        closest[i as usize] = F::min(closest[i as usize], dist);
    }
}
