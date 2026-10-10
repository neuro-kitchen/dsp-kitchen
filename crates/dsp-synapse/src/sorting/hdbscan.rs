//! HDBSCAN density clustering (Campello, Moulavi & Sander 2013; McInnes, Healy & Astels 2017) with
//! the defaults of `sklearn.cluster.HDBSCAN`: Euclidean metric, `min_samples = min_cluster_size`,
//! excess-of-mass cluster selection, no single root cluster. Label `-1` = noise.
//!
//! 1. **Nearest neighbours** of every point, on the device: the `min_samples + 1` nearest (the
//!    point itself counts as the first) with their indices; the core distance is the
//!    `min_samples`-th. Every pass over all pairs shares the points in tiles through shared memory
//!    and is split into launches whose length is measured and kept near [`TARGET_LAUNCH_SECONDS`]
//!    (no launch runs long enough for a display driver to stop it; progress moves after each).
//! 2. **Minimum spanning tree** of the mutual-reachability distance `max(core_a, core_b, ‖a − b‖)`
//!    by Borůvka: every round, each component's cheapest edge (ties by the lower, then the higher
//!    point index, so the tree is the unique minimum one), merged on the host. A full round
//!    compares every point with every other (`O(n²·d)`); two exact shortcuts skip most of it:
//!    - *Neighbour edges.* A point's mutual-reachability weight to anything is at least its own
//!      core distance. When one of its listed neighbours lies in another component and has a core
//!      distance no larger than its own, that edge reaches the bound: it is the point's cheapest,
//!      and (the list holding every point within the core distance) the lowest index among equal
//!      ones. Such points need no scan.
//!    - *Component bound.* Any neighbour edge in another component is a real edge, so the lightest
//!      one bounds the component's cheapest edge from above; a point whose core distance exceeds
//!      that bound cannot improve it, and is not scanned either.
//!
//!    Only the remaining points are scanned against all others. The result is the same tree as a
//!    full scan (`tests::pruning_is_exact`).
//! 3. On the host: single-linkage hierarchy from the sorted edges, condensed at
//!    `min_cluster_size` (`λ = 1 / distance`), cluster stabilities `Σ (λ_point − λ_birth)`, and
//!    excess-of-mass selection.
//!
//! Distances are compared squared throughout.

use std::time::Instant;

use cubecl::prelude::*;
use dsp_base::core::buffer;
use dsp_core::compute::bench::sync;
use dsp_core::compute::LaunchGeometry;

use super::kernels::hdbscan::{cheapest_edge_tile_kernel, knn_candidate_kernel, knn_tile_kernel};
use super::points::DevicePoints;

/// `(point, point, feature)` terms of the first launch of a pass; later launches are sized from
/// the measured speed ([`TARGET_LAUNCH_SECONDS`]).
pub const PAIR_TERMS_PER_LAUNCH: u64 = 1 << 30;

/// Wall time each launch of a pass aims for: long enough that the host wait after it is a small
/// share, short enough for a display driver's watchdog (2 s on Windows) and for visible progress.
pub const TARGET_LAUNCH_SECONDS: f64 = 0.25;

/// Shared memory per unit of the tiled kernels: a point's `d` features, plus its core distance
/// and component (`f32`, `u32`) for the Borůvka step.
fn shared_bytes_per_unit(d: usize) -> usize {
    (d + 2) * size_of::<f32>()
}

/// Most Borůvka rounds over `n` points: each round at least halves the number of components.
fn max_rounds(n: usize) -> usize {
    (usize::BITS - n.saturating_sub(1).leading_zeros()) as usize
}

/// Progress total of [`hdbscan_points_with_progress`] over `n` points: `n` units per pass (the
/// neighbour pass and at most `max_rounds` Borůvka rounds). A launch over `c` of the `n` targets
/// for `r` of the `n` points advances `c · r / n`; a round's skipped work is credited when it ends.
pub fn hdbscan_progress_total(n: usize) -> u64 {
    (1 + max_rounds(n) as u64) * n as u64
}

/// Splits a pass over the `n` target points into launches of a multiple of `units` points,
/// resized after every launch towards [`TARGET_LAUNCH_SECONDS`].
struct Launches {
    n: usize,
    units: usize,
    chunk: usize,
}

impl Launches {
    fn new(n: usize, rows: usize, d: usize, units: usize) -> Self {
        let first = (PAIR_TERMS_PER_LAUNCH / (rows.max(1) as u64 * d.max(1) as u64)).max(1) as usize;
        Self { n, units, chunk: Self::round(first, units, n) }
    }

    fn round(points: usize, units: usize, n: usize) -> usize {
        points.div_ceil(units).max(1).saturating_mul(units).min(n.div_ceil(units) * units)
    }

    /// Runs `launch(j0, j1)` over `0..n` in chunks, waiting for each, calling `step(j1 − j0)`.
    fn run(&mut self, client: &Client, mut launch: impl FnMut(usize, usize), mut step: impl FnMut(usize)) {
        let mut j0 = 0;
        while j0 < self.n {
            let j1 = (j0 + self.chunk).min(self.n);
            let start = Instant::now();
            launch(j0, j1);
            // Each launch finishes before the next: progress is real, and no queue of long launches
            sync(client);
            let seconds = start.elapsed().as_secs_f64();
            if seconds > 0.0 {
                let scaled = (self.chunk as f64 * TARGET_LAUNCH_SECONDS / seconds) as usize;
                self.chunk = Self::round(scaled, self.units, self.n);
            }
            step(j1 - j0);
            j0 = j1;
        }
    }
}

/// Labels of [`hdbscan_points`] for host points `x` (`[n, d]` row-major), uploaded once.
pub fn hdbscan(client: &Client, x: &[f32], n: usize, d: usize, min_cluster_size: usize) -> Vec<i32> {
    hdbscan_points(client, &DevicePoints::upload(client, x, n, d), min_cluster_size)
}

/// Labels of HDBSCAN on device points: cluster index per point (`0..`), `-1` for noise.
pub fn hdbscan_points(client: &Client, points: &DevicePoints, min_cluster_size: usize) -> Vec<i32> {
    hdbscan_points_with_progress(client, points, min_cluster_size, &mut |_, _| {})
}

/// [`hdbscan_points`], calling `progress(done, total)` as the work advances (`total` from
/// [`hdbscan_progress_total`]; the last call is `(total, total)`).
pub fn hdbscan_points_with_progress(
    client: &Client,
    points: &DevicePoints,
    min_cluster_size: usize,
    progress: &mut dyn FnMut(u64, u64),
) -> Vec<i32> {
    spanning_tree_labels(client, points, min_cluster_size, false, true, progress)
}

/// [`hdbscan`] with the `hdbscan` package's `allow_single_cluster` (SpikeInterface's splits set it):
/// excess-of-mass selection may also keep the root, i.e. the points as one cluster. When it does,
/// a point is labelled 0 only if it leaves the root at the root's largest `λ` (the package's
/// labelling), else noise.
pub fn hdbscan_allow_single(client: &Client, x: &[f32], n: usize, d: usize, min_cluster_size: usize) -> Vec<i32> {
    let points = DevicePoints::upload(client, x, n, d);
    spanning_tree_labels(client, &points, min_cluster_size, true, true, &mut |_, _| {})
}

/// Steps 1–3; `prune = false` scans every point in every round (the reference the shortcuts are
/// tested against).
fn spanning_tree_labels(
    client: &Client,
    points: &DevicePoints,
    min_cluster_size: usize,
    allow_single: bool,
    prune: bool,
    progress: &mut dyn FnMut(u64, u64),
) -> Vec<i32> {
    let (n, d) = (points.n, points.d);
    let mcs = min_cluster_size.max(2);
    if n < mcs {
        return vec![-1; n];
    }
    let total = hdbscan_progress_total(n);
    let mut done = 0u64;
    let mut report = |done: u64| progress(done.min(total), total);

    // 1. Nearest neighbours: `list` = min_samples + 1 entries, core distance at min_samples − 1
    let list = mcs + 1;
    let best_d = buffer::filled::<f32>(client, n * list, f32::MAX);
    let best_j = buffer::zeros::<u32>(client, n * list);
    let core = buffer::empty::<f32>(client, n);
    let geom = LaunchGeometry::tiles(client, n, shared_bytes_per_unit(d));
    let units = geom.cube_dim.x;
    Launches::new(n, n, d, units as usize).run(
        client,
        |j0, j1| {
            // SAFETY: `points` holds `d · n`, `best_d` / `best_j` `n · list`, `core` `n` values
            unsafe {
                knn_tile_kernel::launch::<f32>(
                    client,
                    geom.cube_count.clone(),
                    geom.cube_dim.clone(),
                    BufferArg::from_raw_parts(points.handle.clone(), d * n),
                    BufferArg::from_raw_parts(best_d.clone(), n * list),
                    BufferArg::from_raw_parts(best_j.clone(), n * list),
                    BufferArg::from_raw_parts(core.clone(), n),
                    n as u32,
                    j0 as u32,
                    j1 as u32,
                    d as u32,
                    list as u32,
                    (mcs - 1) as u32,
                    units,
                );
            }
        },
        |c| {
            done += c as u64;
            report(done);
        },
    );
    let core_host = buffer::download::<f32>(client, core.clone());

    // 2. Borůvka over the mutual-reachability graph
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    let (cand_w, cand_j, resolved) =
        (buffer::empty::<f32>(client, n), buffer::empty::<u32>(client, n), buffer::empty::<u32>(client, n));
    let (scan_w, scan_j) = (buffer::empty::<f32>(client, n), buffer::empty::<u32>(client, n));
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);
    let mut component: Vec<u32> = (0..n as u32).collect();
    let mut round = 0u64;
    while edges.len() < n - 1 {
        let components = buffer::upload(client, &component);
        // Neighbour edges (all points, one launch)
        let geom_n = LaunchGeometry::elementwise(client, n);
        // SAFETY: the neighbour lists hold `n · list`, the other buffers `n` values
        unsafe {
            knn_candidate_kernel::launch::<f32>(
                client,
                geom_n.cube_count,
                geom_n.cube_dim,
                BufferArg::from_raw_parts(best_d.clone(), n * list),
                BufferArg::from_raw_parts(best_j.clone(), n * list),
                BufferArg::from_raw_parts(core.clone(), n),
                BufferArg::from_raw_parts(components.clone(), n),
                BufferArg::from_raw_parts(cand_w.clone(), n),
                BufferArg::from_raw_parts(cand_j.clone(), n),
                BufferArg::from_raw_parts(resolved.clone(), n),
                n as u32,
                list as u32,
                mcs as u32,
            );
        }
        let mut w = buffer::download::<f32>(client, cand_w.clone());
        let mut j = buffer::download::<u32>(client, cand_j.clone());
        let settled = buffer::download::<u32>(client, resolved.clone());

        // Points that must be scanned: not settled, and able to beat their component's bound
        let mut bound = vec![f32::INFINITY; n];
        for i in 0..n {
            if (j[i] as usize) < n {
                let c = component[i] as usize;
                bound[c] = bound[c].min(w[i]);
            }
        }
        let active: Vec<u32> = (0..n as u32)
            .filter(|&i| {
                let i = i as usize;
                !prune || (settled[i] == 0 && core_host[i] <= bound[component[i] as usize])
            })
            .collect();
        let m = active.len();
        if m > 0 {
            let active_handle = buffer::upload(client, &active);
            let geom_m = LaunchGeometry::tiles(client, m, shared_bytes_per_unit(d));
            let units_m = geom_m.cube_dim.x;
            Launches::new(n, m, d, units_m as usize).run(
                client,
                |j0, j1| {
                    // SAFETY: as in step 1; `active_handle`, `scan_w`, `scan_j` hold ≥ `m` values
                    unsafe {
                        cheapest_edge_tile_kernel::launch::<f32>(
                            client,
                            geom_m.cube_count.clone(),
                            geom_m.cube_dim.clone(),
                            BufferArg::from_raw_parts(points.handle.clone(), d * n),
                            BufferArg::from_raw_parts(core.clone(), n),
                            BufferArg::from_raw_parts(components.clone(), n),
                            BufferArg::from_raw_parts(active_handle.clone(), m),
                            BufferArg::from_raw_parts(scan_w.clone(), m),
                            BufferArg::from_raw_parts(scan_j.clone(), m),
                            n as u32,
                            m as u32,
                            j0 as u32,
                            j1 as u32,
                            u32::from(j0 == 0),
                            d as u32,
                            units_m,
                        );
                    }
                },
                |c| {
                    done += c as u64 * m as u64 / n as u64;
                    report(done);
                },
            );
            let sw = buffer::download_prefix::<f32>(client, scan_w.clone(), m);
            let sj = buffer::download_prefix::<u32>(client, scan_j.clone(), m);
            for (a, &i) in active.iter().enumerate() {
                w[i as usize] = sw[a];
                j[i as usize] = sj[a];
            }
        }
        // The skipped share of this round is done
        round += 1;
        done = round * n as u64 + n as u64;
        report(done);

        // Each component's cheapest edge, under the total order (weight, lower index, higher index)
        let mut cheapest: Vec<Option<(f32, usize, usize)>> = vec![None; n];
        for i in 0..n {
            if j[i] as usize >= n {
                continue;
            }
            let (lo, hi) = (i.min(j[i] as usize), i.max(j[i] as usize));
            let slot = &mut cheapest[component[i] as usize];
            let better = slot.is_none_or(|(bw, blo, bhi)| (w[i], lo, hi) < (bw, blo, bhi));
            if better {
                *slot = Some((w[i], lo, hi));
            }
        }
        let before = edges.len();
        for (weight, a, b) in cheapest.into_iter().flatten() {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
                edges.push((a, b, (weight as f64).sqrt()));
            }
        }
        assert!(edges.len() > before, "hdbscan: Borůvka round added no edge");
        for (i, c) in component.iter_mut().enumerate() {
            *c = root(&mut parent, i) as u32;
        }
    }
    report(total);
    labels_from_spanning_tree(n, mcs, edges, allow_single)
}

/// Step 3: labels from the minimum spanning tree `edges` (`(a, b, distance)`, `n − 1` of them).
fn labels_from_spanning_tree(n: usize, mcs: usize, mut edges: Vec<(usize, usize, f64)>, allow_single: bool) -> Vec<i32> {
    edges.sort_by(|a, b| a.2.total_cmp(&b.2));

    // 3a. Single-linkage hierarchy: node n + i merges two components at edges[i].2
    let total = 2 * n - 1;
    let mut parent: Vec<usize> = (0..total).collect();
    let mut size = vec![1usize; total];
    let mut children = vec![(0usize, 0usize, 0.0f64); n - 1];
    fn find(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    for (i, &(a, b, w)) in edges.iter().enumerate() {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        let node = n + i;
        parent[ra] = node;
        parent[rb] = node;
        size[node] = size[ra] + size[rb];
        children[i] = (ra, rb, w);
    }
    let root = total - 1;
    let leaves_of = |node: usize| -> Vec<usize> {
        let mut stack = vec![node];
        let mut out = Vec::new();
        while let Some(v) = stack.pop() {
            if v < n {
                out.push(v);
            } else {
                let (l, r, _) = children[v - n];
                stack.push(l);
                stack.push(r);
            }
        }
        out
    };

    // 3b. Condensed tree: (parent cluster, child cluster or point, λ, size)
    let mut cluster_of_node = vec![usize::MAX; total];
    cluster_of_node[root] = 0;
    let mut n_clusters = 1usize;
    let mut cluster_rows: Vec<(usize, usize, f64, usize)> = Vec::new(); // parent, child cluster, λ, size
    let mut point_rows: Vec<(usize, usize, f64)> = Vec::new(); // parent cluster, point, λ
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node < n {
            continue;
        }
        let c = cluster_of_node[node];
        let (l, r, w) = children[node - n];
        let lambda = if w > 0.0 { 1.0 / w } else { f64::INFINITY };
        let (big_l, big_r) = (size[l] >= mcs, size[r] >= mcs);
        let drop_points = |sub: usize, rows: &mut Vec<(usize, usize, f64)>| {
            for p in leaves_of(sub) {
                rows.push((c, p, lambda));
            }
        };
        match (big_l, big_r) {
            (true, true) => {
                for child in [l, r] {
                    cluster_of_node[child] = n_clusters;
                    cluster_rows.push((c, n_clusters, lambda, size[child]));
                    n_clusters += 1;
                    stack.push(child);
                }
            }
            (false, false) => {
                drop_points(l, &mut point_rows);
                drop_points(r, &mut point_rows);
            }
            (true, false) => {
                cluster_of_node[l] = c;
                drop_points(r, &mut point_rows);
                stack.push(l);
            }
            (false, true) => {
                cluster_of_node[r] = c;
                drop_points(l, &mut point_rows);
                stack.push(r);
            }
        }
    }

    // Stabilities
    let mut birth = vec![0.0f64; n_clusters];
    let mut cluster_parent = vec![usize::MAX; n_clusters];
    for &(p, c, lambda, _) in &cluster_rows {
        birth[c] = lambda;
        cluster_parent[c] = p;
    }
    let mut stability = vec![0.0f64; n_clusters];
    for &(p, _, lambda) in &point_rows {
        stability[p] += lambda.min(f64::MAX) - birth[p];
    }
    for &(p, _, lambda, sz) in &cluster_rows {
        stability[p] += (lambda - birth[p]) * sz as f64;
    }

    // 3c. Excess of mass: children are created after their parent, so walk ids downwards
    let mut selected = vec![true; n_clusters];
    selected[0] = allow_single; // the root is a candidate only with `allow_single`
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for &(p, c, _, _) in &cluster_rows {
        kids[p].push(c);
    }
    let mut subtree = stability.clone();
    let first = if allow_single { 0 } else { 1 };
    for c in (first..n_clusters).rev() {
        let kid_sum: f64 = kids[c].iter().map(|&k| subtree[k]).sum();
        if !kids[c].is_empty() && kid_sum > stability[c] {
            selected[c] = false;
            subtree[c] = kid_sum;
        } else {
            // Keep `c`; deselect everything below it
            let mut stack = kids[c].clone();
            while let Some(k) = stack.pop() {
                selected[k] = false;
                stack.extend(kids[k].iter().copied());
            }
        }
    }

    // Labels: each point belongs to the selected cluster on its path to the root
    let order: Vec<usize> = (0..n_clusters).filter(|&c| selected[c]).collect();
    let mut label_of_cluster = vec![-1i32; n_clusters];
    for (i, &c) in order.iter().enumerate() {
        label_of_cluster[c] = i as i32;
    }
    let mut labels = vec![-1i32; n];
    // The root as the only cluster: only the points leaving it at its largest λ keep the label
    let root_max = point_rows.iter().filter(|r| r.0 == 0).map(|r| r.2).chain(cluster_rows.iter().filter(|r| r.0 == 0).map(|r| r.2)).fold(f64::NEG_INFINITY, f64::max);
    for &(p, point, lambda) in &point_rows {
        let mut c = p;
        while c != usize::MAX {
            if selected[c] {
                if c != 0 || lambda >= root_max {
                    labels[point] = label_of_cluster[c];
                }
                break;
            }
            c = cluster_parent[c];
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::{ComputeTarget, ComputeTask};

    /// Relative difference under which two `f64` edge weights count as tied (the device computes
    /// them in `f32`).
    const TIE_TOLERANCE: f64 = 1e-6;

    /// Steps 1–2 on the host (Prim, `f64`): the reference the device path is checked against.
    fn spanning_tree_host(x: &[f32], n: usize, d: usize, mcs: usize) -> Vec<(usize, usize, f64)> {
        let dist = |a: usize, b: usize| -> f64 {
            x[a * d..(a + 1) * d].iter().zip(&x[b * d..(b + 1) * d]).map(|(p, q)| ((p - q) as f64).powi(2)).sum::<f64>().sqrt()
        };

        // 1. Core distances (min_samples = min_cluster_size, self included)
        let min_samples = mcs;
        let core: Vec<f64> = (0..n)
            .map(|i| {
                let mut ds: Vec<f64> = (0..n).map(|j| dist(i, j)).collect();
                let kth = (min_samples - 1).min(n - 1);
                *ds.select_nth_unstable_by(kth, f64::total_cmp).1
            })
            .collect();
        let mreach = |a: usize, b: usize| dist(a, b).max(core[a]).max(core[b]);

        // 2. Prim's MST over mutual reachability
        let mut in_tree = vec![false; n];
        let mut best = vec![f64::INFINITY; n];
        let mut from = vec![0usize; n];
        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);
        let mut current = 0usize;
        in_tree[0] = true;
        for _ in 1..n {
            let mut next = usize::MAX;
            let mut next_d = f64::INFINITY;
            for j in 0..n {
                if in_tree[j] {
                    continue;
                }
                let dj = mreach(current, j);
                if dj < best[j] {
                    best[j] = dj;
                    from[j] = current;
                }
                if best[j] < next_d {
                    next_d = best[j];
                    next = j;
                }
            }
            in_tree[next] = true;
            edges.push((from[next], next, next_d));
            current = next;
        }
        edges
    }

    fn blobs() -> Vec<f32> {
        let mut x = Vec::new();
        for (cx, cy) in [(0.0f32, 0.0f32), (20.0, 0.0)] {
            for i in 0..25 {
                let a = i as f32 * 0.7;
                x.extend_from_slice(&[cx + a.cos() * (0.2 + 0.02 * i as f32), cy + a.sin() * 0.3]);
            }
        }
        x.extend_from_slice(&[100.0, 100.0]);
        x
    }

    struct Labels(Vec<f32>, usize, usize, usize);
    impl ComputeTask for Labels {
        type Output = Vec<i32>;
        fn run(self, client: Client) -> Self::Output {
            hdbscan(&client, &self.0, self.1, self.2, self.3)
        }
    }

    #[test]
    fn allow_single_keeps_one_blob_whole() {
        // One elongated blob: without the option HDBSCAN must split it; with it, the root may stay
        let mut state = 3u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
        };
        let n = 400;
        let x: Vec<f32> = (0..n).flat_map(|_| [rnd(), rnd()]).collect();
        let tree = super::tests::spanning_tree_host(&x, n, 2, 20);
        let single = labels_from_spanning_tree(n, 20, tree.clone(), true);
        let k_single = single.iter().copied().max().unwrap_or(-1) + 1;
        let k = labels_from_spanning_tree(n, 20, tree, false).iter().copied().max().unwrap_or(-1) + 1;
        assert!(k_single <= 1, "uniform square kept as at most one cluster: {k_single}");
        assert!(k != 1, "without the option the root is never chosen: {k}");
    }

    #[test]
    fn two_blobs_and_an_outlier() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let labels = target.run(Labels(blobs(), 51, 2, 5)).expect("target run");
        assert_eq!(labels[50], -1, "outlier is noise");
        let (a, b) = (labels[0], labels[25]);
        assert!(a >= 0 && b >= 0 && a != b, "{labels:?}");
        assert!(labels[..25].iter().all(|&l| l == a || l == -1));
        assert!(labels[25..50].iter().all(|&l| l == b || l == -1));
    }

    /// The shortcuts of step 2 give the same labels as scanning every point in every round, on
    /// clip-like data with clusters of different spreads, scattered points and exact duplicates
    /// (distance ties).
    #[test]
    fn pruning_is_exact() {
        struct Both(Vec<f32>, usize, usize, usize);
        impl ComputeTask for Both {
            type Output = (Vec<i32>, Vec<i32>);
            fn run(self, client: Client) -> Self::Output {
                let points = DevicePoints::upload(&client, &self.0, self.1, self.2);
                let pruned = spanning_tree_labels(&client, &points, self.3, false, true, &mut |_, _| {});
                let full = spanning_tree_labels(&client, &points, self.3, false, false, &mut |_, _| {});
                (pruned, full)
            }
        }
        let Ok(target) = ComputeTarget::from_env() else { return };
        let (n, d, mcs) = (3_000usize, 61usize, 20usize);
        let mut state = 0x2545_f491u32;
        let mut uniform = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let mut x = vec![0.0f32; n * d];
        for i in 0..n {
            let row = i * d;
            if i % 97 == 96 {
                x.copy_within(row - d..row, row);
                continue;
            }
            let (k, spread, amp) = if i % 23 == 0 { (0, 6.0, 0.0) } else { (i % 9, 0.1 + 0.08 * (i % 9) as f32, 1.0 + 0.3 * (i % 9) as f32) };
            for t in 0..d {
                let tau = (t as f32 - 20.0 - k as f32) / (2.0 + k as f32 * 0.5);
                x[row + t] = -amp * (-tau * tau).exp() + spread * uniform();
            }
        }
        let (pruned, full) = target.run(Both(x, n, d, mcs)).expect("target run");
        assert!(full.iter().any(|&l| l >= 0), "the data form clusters");
        assert_eq!(pruned, full, "pruned and full Borůvka disagree");
    }

    #[test]
    fn device_matches_host_reference() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        // Three blobs of different spreads in 5 dimensions, plus scattered points
        let (n, d, mcs) = (240usize, 5usize, 8usize);
        let mut state = 0x9e37_79b9u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let x: Vec<f32> = (0..n)
            .flat_map(|i| {
                let (centre, spread) = match i % 4 {
                    0 => (0.0, 1.0),
                    1 => (10.0, 0.5),
                    2 => (-12.0, 2.0),
                    _ => (0.0, 40.0),
                };
                (0..d).map(|_| centre + spread * next()).collect::<Vec<_>>()
            })
            .collect();
        let host_edges = spanning_tree_host(&x, n, d, mcs);
        let want = labels_from_spanning_tree(n, mcs, host_edges, false);
        let got = target.run(Labels(x.clone(), n, d, mcs)).expect("target run");
        // Equal mutual-reachability weights are common (`max(core_i, core_j, d) = core_i` for
        // every close neighbour with a smaller core distance): Prim (the host) and Borůvka (the
        // device) then pick different, equally minimal edges, and a point attached through a tie
        // may end up in a cluster in one tree and noise in the other. Every other point must agree.
        let tied = |p: usize| {
            let dist = |a: usize, b: usize| x[a * d..(a + 1) * d].iter().zip(&x[b * d..(b + 1) * d]).map(|(u, v)| ((u - v) as f64).powi(2)).sum::<f64>();
            let core = |a: usize| {
                let mut ds: Vec<f64> = (0..n).map(|b| dist(a, b)).collect();
                *ds.select_nth_unstable_by(mcs - 1, f64::total_cmp).1
            };
            let weights: Vec<f64> = (0..n).filter(|&j| j != p).map(|j| dist(p, j).max(core(p)).max(core(j))).collect();
            let lightest = weights.iter().copied().fold(f64::INFINITY, f64::min);
            weights.iter().filter(|&&w| w <= lightest * (1.0 + TIE_TOLERANCE)).count() > 1
        };
        for p in (0..n).filter(|&p| got[p] != want[p]) {
            assert!(tied(p), "point {p}: device label {} vs host {} without a tie", got[p], want[p]);
        }
    }
}
