//! HDBSCAN density clustering (Campello, Moulavi & Sander 2013; McInnes, Healy & Astels 2017) with
//! the defaults of `sklearn.cluster.HDBSCAN`: Euclidean metric, `min_samples = min_cluster_size`,
//! excess-of-mass cluster selection, no single root cluster. Label `-1` = noise.
//!
//! 1. Core distance of each point, on the device: distance to its `min_samples`-th nearest
//!    neighbour (the point itself counts as the first), the nearest kept in registers.
//! 2. Minimum spanning tree of the mutual-reachability distance `max(core_a, core_b, ‖a − b‖)`, on
//!    the device by Borůvka: every round, each point's cheapest edge to another component
//!    (`O(n²·d)`, one unit per point); the host keeps each component's cheapest (ties by the
//!    lower, then the higher point index, so the tree is the unique minimum one) and merges.
//!    About `log₂ n` rounds, each reading back `O(n)` values. Distances are compared squared.
//! 3. On the host: single-linkage hierarchy from the sorted edges, condensed at
//!    `min_cluster_size` (`λ = 1 / distance`), cluster stabilities `Σ (λ_point − λ_birth)`, and
//!    excess-of-mass selection.

use cubecl::prelude::*;
use dsp_base::core::buffer;
use dsp_core::compute::LaunchGeometry;

use super::kernels::points::{cheapest_edge_kernel, core_distance_kernel};
use super::points::DevicePoints;

/// Labels of [`hdbscan_points`] for host points `x` (`[n, d]` row-major), uploaded once.
pub fn hdbscan<R: Runtime>(client: &ComputeClient<R>, x: &[f32], n: usize, d: usize, min_cluster_size: usize) -> Vec<i32> {
    hdbscan_points(client, &DevicePoints::upload(client, x, n, d), min_cluster_size)
}

/// Labels of HDBSCAN on device points: cluster index per point (`0..`), `-1` for noise.
pub fn hdbscan_points<R: Runtime>(client: &ComputeClient<R>, points: &DevicePoints, min_cluster_size: usize) -> Vec<i32> {
    let (n, d) = (points.n, points.d);
    let mcs = min_cluster_size.max(2);
    if n < mcs {
        return vec![-1; n];
    }
    let geom = LaunchGeometry::elementwise(client, n);

    // 1. Squared core distances (min_samples = min_cluster_size, self included)
    let core = buffer::empty::<R, f32>(client, n);
    // SAFETY: `points` holds `d · n` values, `core` `n`
    unsafe {
        core_distance_kernel::launch::<f32, R>(
            client,
            geom.cube_count.clone(),
            geom.cube_dim.clone(),
            ArrayArg::from_raw_parts(points.handle.clone(), d * n),
            ArrayArg::from_raw_parts(core.clone(), n),
            n as u32,
            d as u32,
            mcs as u32,
        );
    }

    // 2. Borůvka over the mutual-reachability graph
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    let (best_w, best_j) = (buffer::empty::<R, f32>(client, n), buffer::empty::<R, u32>(client, n));
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);
    let mut component: Vec<u32> = (0..n as u32).collect();
    while edges.len() < n - 1 {
        // SAFETY: as above; `component`, `best_w`, `best_j` hold `n` values
        unsafe {
            cheapest_edge_kernel::launch::<f32, R>(
                client,
                geom.cube_count.clone(),
                geom.cube_dim.clone(),
                ArrayArg::from_raw_parts(points.handle.clone(), d * n),
                ArrayArg::from_raw_parts(core.clone(), n),
                ArrayArg::from_raw_parts(buffer::upload(client, &component), n),
                ArrayArg::from_raw_parts(best_w.clone(), n),
                ArrayArg::from_raw_parts(best_j.clone(), n),
                n as u32,
                d as u32,
            );
        }
        let w = buffer::download_prefix::<R, f32>(client, best_w.clone(), n);
        let j = buffer::download_prefix::<R, u32>(client, best_j.clone(), n);
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
    labels_from_spanning_tree(n, mcs, edges)
}

/// Step 3: labels from the minimum spanning tree `edges` (`(a, b, distance)`, `n − 1` of them).
fn labels_from_spanning_tree(n: usize, mcs: usize, mut edges: Vec<(usize, usize, f64)>) -> Vec<i32> {
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
    selected[0] = false; // no single root cluster
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for &(p, c, _, _) in &cluster_rows {
        kids[p].push(c);
    }
    let mut subtree = stability.clone();
    for c in (1..n_clusters).rev() {
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
    for &(p, point, _) in &point_rows {
        let mut c = p;
        while c != usize::MAX {
            if selected[c] {
                labels[point] = label_of_cluster[c];
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
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
            hdbscan(&client, &self.0, self.1, self.2, self.3)
        }
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
        let host_weight: f64 = host_edges.iter().map(|e| e.2).sum();
        let want = labels_from_spanning_tree(n, mcs, host_edges);
        let got = target.run(Labels(x, n, d, mcs)).expect("target run");
        assert_eq!(got, want, "labels (host tree weight {host_weight})");
    }
}
