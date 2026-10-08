//! Graph-based clustering on a bipartite k-nearest-neighbour graph, on the device: the scalable
//! modularity clustering of the Kilosort4 paper (Pachitariu et al. 2024, Methods, *Scaling up the
//! graph-based clustering*), written from its description.
//!
//! 1. **Graph.** Left nodes are all `n` points; right nodes are a subset (every `subset_stride`-th
//!    point, at most `max_subset`, so the neighbourhood scale does not shrink as recordings grow).
//!    Each left node links to its `neighbours` nearest right nodes (exact squared distances,
//!    `‖x‖² + ‖y‖² − 2·X·Yᵀ` through [`dsp_base::linalg::matmul`], in row chunks; a point is not its
//!    own neighbour). Every edge joins a left and a right node, so given the right labels every left
//!    node can be assigned independently, and vice versa: the steps are fully parallel.
//! 2. **Initialization.** `init_clusters` seeds by greedy k-means++ (each seed the best of
//!    `2 + ln k` candidates drawn by squared distance), every point labelled by its nearest seed.
//! 3. **Iteration.** Right nodes, then left nodes, each take the cluster among their neighbours'
//!    that maximizes the paper's bipartite modularity gain `n_tc − γ · k_t · K_c / 2m` (`n_tc`
//!    neighbours of `t` in `c`, `k_t` its degree, `K_c` the degree sum of the other side in `c`,
//!    `m` edges); until no left node moves, at most `max_iterations` rounds.
//! 4. **Result.** Left labels compacted to `0..n_clusters` (kept on the device) and the edge counts
//!    between clusters, from which merge trees are built.
//!
//! Choices of ours where the paper is silent: a node only joins a cluster among its neighbours'
//! (as Louvain-type methods do), ties go to the smallest label, iteration stops at convergence.
//!
//! Data movement: the points are on the device already; k-means++ uploads its uniforms once and
//! downloads its seeds once; per round one `u32` (moved nodes) comes back; at the end the cluster
//! sizes and the `[c, c]` edge counts. Labels stay on the device.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::linalg::{matmul, MatrixView};
use dsp_core::compute::LaunchGeometry;

use super::kernels::bipartite::{
    closest_update_chosen_kernel, count_values_kernel, cluster_edges_kernel, degree_sums_kernel, draw_candidates_kernel, left_assign_kernel,
    pick_seed_kernel, relabel_kernel, right_assign_kernel, right_counts_kernel, select_neighbours_kernel, sq_norms_kernel,
    trial_potentials_kernel, NO_LABEL,
};
use super::kernels::points::block_sums_kernel;
use super::kmeans::Rng;
use super::points::DevicePoints;

/// Inner products computed per chunk of rows (`rows · m` values): bounds the scratch buffer.
const MAX_CHUNK_DOTS: usize = 1 << 25;
/// Rows of a chunk are a multiple of this (keeps every chunk's start aligned for binding).
const CHUNK_ROWS_ALIGN: usize = 64;
/// Points per partial sum of the k-means++ trial potentials and per block of the draws. Each unit
/// walks its block serially, so small blocks keep the chains short: seeding 200 seeds of 1 416
/// points × 102 features took 3.4 s with 1024, 241 ms with 64, 76 ms with 16 (RTX 2070; then
/// launch overhead dominates). Measured by `time_seeding_blocks`.
const TRIAL_BLOCK: usize = 16;

/// Settings of [`bipartite_clustering`] (defaults: the Kilosort4 paper's).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BipartiteOptions {
    /// Right neighbours of every left node.
    pub neighbours: usize,
    /// Every `subset_stride`-th point is a right node.
    pub subset_stride: usize,
    /// At most this many right nodes (evenly thinned).
    pub max_subset: usize,
    /// k-means++ seeds the iteration starts from.
    pub init_clusters: usize,
    pub max_iterations: usize,
    /// Modularity resolution γ.
    pub resolution: f64,
    /// Seed of the k-means++ draws.
    pub seed: u64,
}

impl Default for BipartiteOptions {
    fn default() -> Self {
        Self { neighbours: 10, subset_stride: 20, max_subset: 25_000, init_clusters: 200, max_iterations: 200, resolution: 1.0, seed: 0 }
    }
}

/// Result of [`bipartite_clustering`].
pub struct BipartiteClustering {
    pub n_clusters: usize,
    /// `[n]` cluster of every point (`u32`, `0..n_clusters`), on the device.
    pub labels: Handle,
    /// Points in each cluster.
    pub sizes: Vec<usize>,
    /// `[n_clusters, n_clusters]`: edges from left nodes of cluster `i` to right nodes of `j`.
    pub edges: Vec<u32>,
    /// Assignment rounds run.
    pub iterations: usize,
}

/// Right nodes: every `stride`-th point, evenly thinned to at most `max`.
pub fn subset_indices(n: usize, stride: usize, max: usize) -> Vec<u32> {
    let all: Vec<u32> = (0..n).step_by(stride.max(1)).map(|i| i as u32).collect();
    if all.len() <= max.max(1) {
        return all;
    }
    let keep = max.max(1);
    (0..keep).map(|j| all[j * all.len() / keep]).collect()
}

fn launch_norms(client: &Client, points: &DevicePoints) -> Handle {
    let (n, d) = (points.n, points.d);
    let out = buffer::empty::<f32>(client, n);
    let geom = LaunchGeometry::elementwise(client, n);
    // SAFETY: `points` holds `d · n` values, `out` `n`
    unsafe {
        sq_norms_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(points.handle.clone(), d * n),
            BufferArg::from_raw_parts(out.clone(), n),
            n as u32,
            d as u32,
        );
    }
    out
}

/// The `k` nearest `right` points of every `left` point (`[n_left · k]` indices into `right`,
/// nearest first). `right_ids[j]` is right point `j`'s index among the left points, so a point is
/// not its own neighbour ([`NO_LABEL`] for right points that are none of them). Needs
/// `k ≤ right.n` (`k < right.n` when the right points are a subset of the left ones).
pub fn nearest_neighbours(client: &Client, left: &DevicePoints, right: &DevicePoints, right_ids: &[u32], k: usize) -> Handle {
    let (n, m, d) = (left.n, right.n, left.d);
    assert_eq!(right.d, d, "nearest_neighbours: feature counts differ");
    assert_eq!(right_ids.len(), m, "nearest_neighbours: one id per right point");
    assert!(k >= 1 && k <= m, "nearest_neighbours: need 1 ≤ k ≤ right points");
    let nb = buffer::empty::<u32>(client, n * k);
    let (left_norms, right_norms) = (launch_norms(client, left), launch_norms(client, right));
    let ids = buffer::upload(client, right_ids);
    let chunk = ((MAX_CHUNK_DOTS / m.max(1)) / CHUNK_ROWS_ALIGN * CHUNK_ROWS_ALIGN).max(CHUNK_ROWS_ALIGN).min(n.next_multiple_of(CHUNK_ROWS_ALIGN));
    let dots = buffer::empty::<f32>(client, chunk * m);
    // X is the transpose of the feature-major `[d, n]` buffer; Yᵀ is the right points as stored
    let lhs_all = MatrixView::row_major(&left.handle, d * n, d, n);
    let rhs = MatrixView::row_major(&right.handle, d * m, d, m);
    let mut row0 = 0;
    while row0 < n {
        let rows = chunk.min(n - row0);
        let lhs = lhs_all.columns(row0..row0 + rows).transposed();
        matmul::<f32>(client, &lhs, &rhs, &dots, rows * m);
        let geom = LaunchGeometry::elementwise(client, rows);
        // SAFETY: `dots` holds `rows · m`, the norms `n` / `m`, `ids` `m`, `nb` `n · k` values
        unsafe {
            select_neighbours_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(dots.clone(), rows * m),
                BufferArg::from_raw_parts(left_norms.clone(), n),
                BufferArg::from_raw_parts(right_norms.clone(), m),
                BufferArg::from_raw_parts(ids.clone(), m),
                BufferArg::from_raw_parts(nb.clone(), n * k),
                row0 as u32,
                rows as u32,
                m as u32,
                k as u32,
            );
        }
        row0 += rows;
    }
    nb
}

/// Greedy k-means++ seeds: per seed, `2 + ln k` candidates drawn by squared distance to the nearest
/// seed, all scored in one launch, the lowest potential kept (the first on ties). Everything stays on
/// the device: the uniforms of every draw are made on the host and uploaded once, each seed is five
/// queued launches (block sums, draws, potentials, pick, distance update) and the `k` seed indices
/// come back once at the end. Where every point already coincides with a seed, later draws repeat
/// points (their clusters stay empty and are dropped by the compaction).
fn kmeans_plus_plus_seeds(client: &Client, points: &DevicePoints, k: usize, rng: &mut Rng) -> Vec<u32> {
    kmeans_plus_plus_seeds_blocked(client, points, k, rng, TRIAL_BLOCK)
}

fn kmeans_plus_plus_seeds_blocked(client: &Client, points: &DevicePoints, k: usize, rng: &mut Rng, block: usize) -> Vec<u32> {
    let (n, d) = (points.n, points.d);
    let trials = 2 + (k as f64).ln() as usize;
    let blocks = n.div_ceil(block);
    let first = (rng.next_u64() % n as u64) as u32;
    let uniforms: Vec<f32> = (0..k * trials).map(|_| rng.uniform() as f32).collect();
    let uniforms = buffer::upload(client, &uniforms);
    let mut init = vec![first; k];
    init.iter_mut().skip(1).for_each(|s| *s = 0);
    let seeds = buffer::upload(client, &init);
    let chosen = buffer::upload(client, &[first]);
    let closest = buffer::upload(client, &vec![f32::MAX; n]);
    let (sums, cand, partial) = (buffer::empty::<f32>(client, blocks), buffer::empty::<u32>(client, trials), buffer::empty::<f32>(client, trials * blocks));
    let per_point = LaunchGeometry::elementwise(client, n);
    let per_block = LaunchGeometry::elementwise(client, blocks);
    let per_trial = LaunchGeometry::elementwise(client, trials);
    let per_partial = LaunchGeometry::elementwise(client, trials * blocks);
    let update = || {
        // SAFETY: `points` holds `d · n`, `closest` `n`, `chosen` one value
        unsafe {
            closest_update_chosen_kernel::launch::<f32>(
                client,
                per_point.cube_count.clone(),
                per_point.cube_dim.clone(),
                BufferArg::from_raw_parts(points.handle.clone(), d * n),
                BufferArg::from_raw_parts(closest.clone(), n),
                BufferArg::from_raw_parts(chosen.clone(), 1),
                n as u32,
                d as u32,
            );
        }
    };
    update();
    for s in 1..k {
        // SAFETY: `closest` holds `n`, `sums` `blocks`, `uniforms` `k · trials`, `cand` `trials`,
        // `partial` `trials · blocks`, `seeds` `k`, `chosen` one value
        unsafe {
            block_sums_kernel::launch::<f32>(
                client,
                per_block.cube_count.clone(),
                per_block.cube_dim.clone(),
                BufferArg::from_raw_parts(closest.clone(), n),
                BufferArg::from_raw_parts(sums.clone(), blocks),
                n as u32,
                block as u32,
                blocks as u32,
            );
            draw_candidates_kernel::launch::<f32>(
                client,
                per_trial.cube_count.clone(),
                per_trial.cube_dim.clone(),
                BufferArg::from_raw_parts(closest.clone(), n),
                BufferArg::from_raw_parts(sums.clone(), blocks),
                BufferArg::from_raw_parts(uniforms.clone(), k * trials),
                BufferArg::from_raw_parts(cand.clone(), trials),
                n as u32,
                block as u32,
                blocks as u32,
                trials as u32,
                s as u32,
            );
            trial_potentials_kernel::launch::<f32>(
                client,
                per_partial.cube_count.clone(),
                per_partial.cube_dim.clone(),
                BufferArg::from_raw_parts(points.handle.clone(), d * n),
                BufferArg::from_raw_parts(closest.clone(), n),
                BufferArg::from_raw_parts(cand.clone(), trials),
                BufferArg::from_raw_parts(partial.clone(), trials * blocks),
                n as u32,
                d as u32,
                trials as u32,
                block as u32,
                blocks as u32,
            );
            pick_seed_kernel::launch::<f32>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(partial.clone(), trials * blocks),
                BufferArg::from_raw_parts(cand.clone(), trials),
                BufferArg::from_raw_parts(seeds.clone(), k),
                BufferArg::from_raw_parts(chosen.clone(), 1),
                trials as u32,
                blocks as u32,
                s as u32,
            );
        }
        update();
    }
    buffer::download_prefix::<u32>(client, seeds, k)
}

/// `counts[v]` of the `n` device values below `bins`.
fn count_values(client: &Client, values: &Handle, n: usize, bins: usize) -> Handle {
    let counts = buffer::zeros::<u32>(client, bins);
    let geom = LaunchGeometry::elementwise(client, n);
    // SAFETY: `values` holds `n`, `counts` `bins` values
    unsafe {
        count_values_kernel::launch(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(values.clone(), n),
            BufferArg::from_raw_parts(counts.clone(), bins),
            n as u32,
            bins as u32,
        );
    }
    counts
}

/// Clusters `points` (see the module docs).
pub fn bipartite_clustering(client: &Client, points: &DevicePoints, opts: &BipartiteOptions) -> BipartiteClustering {
    let n = points.n;
    let subset = subset_indices(n, opts.subset_stride, opts.max_subset);
    let m = subset.len();
    let k = opts.neighbours.min(m.saturating_sub(1));
    if n < 2 || k == 0 {
        let nc = 1.min(n);
        return BipartiteClustering { n_clusters: nc, labels: buffer::zeros::<u32>(client, n.max(1)), sizes: vec![n; nc], edges: vec![0; nc], iterations: 0 };
    }
    let mut rng = Rng(opts.seed);

    // 1. Graph, and the degree of every right node
    let right_points = points.gather(client, &subset);
    let nb = nearest_neighbours(client, points, &right_points, &subset, k);
    let edges_n = n * k;
    let right_degree = count_values(client, &nb, edges_n, m);

    // 2. Initial labels: nearest k-means++ seed (no point is excluded from its nearest seed)
    let seeds = kmeans_plus_plus_seeds(client, points, opts.init_clusters.min(n).max(1), &mut rng);
    let c = seeds.len();
    let left = nearest_neighbours(client, points, &points.gather(client, &seeds), &vec![NO_LABEL; c], 1);
    let right = buffer::upload(client, &vec![NO_LABEL; m]);

    // 3. Alternating right / left assignment
    let penalty = opts.resolution / (2.0 * edges_n as f64);
    let per_edge = LaunchGeometry::elementwise(client, edges_n);
    let per_right = LaunchGeometry::elementwise(client, m);
    let per_left = LaunchGeometry::elementwise(client, n);
    let mut iterations = 0;
    while iterations < opts.max_iterations {
        iterations += 1;
        let k_left = count_values(client, &left, n, c);
        let counts = buffer::zeros::<u32>(client, m * c);
        let k_right = buffer::zeros::<u32>(client, c);
        let changed = buffer::zeros::<u32>(client, 1);
        // SAFETY: `nb` holds `n · k`, `left` `n`, `right` and `right_degree` `m`, `counts` `m · c`,
        // `k_left` / `k_right` `c` values
        unsafe {
            right_counts_kernel::launch(
                client,
                per_edge.cube_count.clone(),
                per_edge.cube_dim.clone(),
                BufferArg::from_raw_parts(nb.clone(), edges_n),
                BufferArg::from_raw_parts(left.clone(), n),
                BufferArg::from_raw_parts(counts.clone(), m * c),
                edges_n as u32,
                k as u32,
                c as u32,
            );
            // Left nodes all have degree k: their degree sum per cluster is k · their count
            right_assign_kernel::launch::<f32>(
                client,
                per_right.cube_count.clone(),
                per_right.cube_dim.clone(),
                BufferArg::from_raw_parts(counts.clone(), m * c),
                BufferArg::from_raw_parts(right_degree.clone(), m),
                BufferArg::from_raw_parts(k_left.clone(), c),
                BufferArg::from_raw_parts(right.clone(), m),
                m as u32,
                c as u32,
                (penalty * k as f64) as f32,
            );
            degree_sums_kernel::launch(
                client,
                per_right.cube_count.clone(),
                per_right.cube_dim.clone(),
                BufferArg::from_raw_parts(right.clone(), m),
                BufferArg::from_raw_parts(right_degree.clone(), m),
                BufferArg::from_raw_parts(k_right.clone(), c),
                m as u32,
                c as u32,
            );
            left_assign_kernel::launch::<f32>(
                client,
                per_left.cube_count.clone(),
                per_left.cube_dim.clone(),
                BufferArg::from_raw_parts(nb.clone(), edges_n),
                BufferArg::from_raw_parts(right.clone(), m),
                BufferArg::from_raw_parts(k_right.clone(), c),
                BufferArg::from_raw_parts(left.clone(), n),
                BufferArg::from_raw_parts(changed.clone(), 1),
                n as u32,
                c as u32,
                penalty as f32,
                k as u32,
            );
        }
        if buffer::download::<u32>(client, changed)[0] == 0 {
            break;
        }
    }

    // 4. Compact labels (clusters keeping left nodes, in label order) and count edges between them
    let sizes = buffer::download_prefix::<u32>(client, count_values(client, &left, n, c), c);
    let mut map = vec![NO_LABEL; c];
    let mut n_clusters = 0u32;
    for (label, &size) in sizes.iter().enumerate() {
        if size > 0 {
            map[label] = n_clusters;
            n_clusters += 1;
        }
    }
    let nc = n_clusters as usize;
    let map = buffer::upload(client, &map);
    let edges = buffer::zeros::<u32>(client, nc * nc);
    // SAFETY: `left` holds `n`, `right` `m`, `map` `c`, `nb` `n · k`, `edges` `nc · nc` values
    unsafe {
        relabel_kernel::launch(client, per_left.cube_count, per_left.cube_dim, BufferArg::from_raw_parts(left.clone(), n), BufferArg::from_raw_parts(map.clone(), c), n as u32, c as u32);
        relabel_kernel::launch(client, per_right.cube_count, per_right.cube_dim, BufferArg::from_raw_parts(right.clone(), m), BufferArg::from_raw_parts(map, c), m as u32, c as u32);
        cluster_edges_kernel::launch(
            client,
            per_edge.cube_count,
            per_edge.cube_dim,
            BufferArg::from_raw_parts(nb, edges_n),
            BufferArg::from_raw_parts(left.clone(), n),
            BufferArg::from_raw_parts(right, m),
            BufferArg::from_raw_parts(edges.clone(), nc * nc),
            edges_n as u32,
            k as u32,
            nc as u32,
        );
    }
    let edges = buffer::download_prefix::<u32>(client, edges, nc * nc);
    let sizes = sizes.iter().filter(|&&s| s > 0).map(|&s| s as usize).collect();
    BipartiteClustering { n_clusters: nc, labels: left, sizes, edges, iterations }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;

    /// Deterministic points: `blobs` Gaussian-like clusters (sums of uniforms) of `per` points in
    /// `d` dimensions, centres `spread` apart along the first axis.
    fn blobs(blobs: usize, per: usize, d: usize, spread: f32) -> Vec<f32> {
        let mut rng = Rng(7);
        let mut x = Vec::with_capacity(blobs * per * d);
        for b in 0..blobs {
            for _ in 0..per {
                for f in 0..d {
                    let noise: f32 = (0..4).map(|_| rng.uniform() as f32 - 0.5).sum();
                    x.push(noise + if f == 0 { b as f32 * spread } else { 0.0 });
                }
            }
        }
        x
    }

    /// Host reference of the neighbour lists: squared distances in `f64`, self skipped, ties to the
    /// smaller index.
    fn host_neighbours(x: &[f32], n: usize, d: usize, subset: &[u32], k: usize) -> Vec<Vec<u32>> {
        (0..n)
            .map(|i| {
                let mut cand: Vec<(f64, u32)> = subset
                    .iter()
                    .enumerate()
                    .filter(|&(_, &s)| s as usize != i)
                    .map(|(j, &s)| ((0..d).map(|f| ((x[i * d + f] - x[s as usize * d + f]) as f64).powi(2)).sum(), j as u32))
                    .collect();
                cand.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                cand.into_iter().take(k).map(|(_, j)| j).collect()
            })
            .collect()
    }

    #[test]
    #[ignore = "timing: run with --ignored --nocapture"]
    fn time_seeding_blocks() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (n, d, k) = (1416usize, 102usize, 200usize);
        let points = DevicePoints::upload(&client, &blobs(4, n / 4, d, 3.0), n, d);
        for block in [1024usize, 256, 64, 16] {
            let _ = kmeans_plus_plus_seeds_blocked(&client, &points, k, &mut Rng(1), block);
            let t = std::time::Instant::now();
            let _ = kmeans_plus_plus_seeds_blocked(&client, &points, k, &mut Rng(1), block);
            println!("seeding n {n} d {d} k {k} block {block}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
    }

    #[test]
    fn subset_is_strided_then_thinned() {
        assert_eq!(subset_indices(10, 3, 100), vec![0, 3, 6, 9]);
        assert_eq!(subset_indices(100, 1, 4), vec![0, 25, 50, 75]);
    }

    /// The device neighbour lists match the host's (as sets: f32 dot products may swap near-equal
    /// distances), on odd sizes spanning two row chunks' worth of alignment.
    #[test]
    fn neighbours_match_host() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (n, d, k) = (301usize, 13usize, 5usize);
        let x = blobs(3, n / 3 + 1, d, 4.0)[..n * d].to_vec();
        let points = DevicePoints::upload(&client, &x, n, d);
        let subset = subset_indices(n, 3, 1000);
        let nb = nearest_neighbours(&client, &points, &points.gather(&client, &subset), &subset, k);
        let got = buffer::download_prefix::<u32>(&client, nb, n * k);
        let want = host_neighbours(&x, n, d, &subset, k);
        let mut differ = 0;
        for i in 0..n {
            let mut a = got[i * k..(i + 1) * k].to_vec();
            let mut b = want[i].clone();
            assert!(!a.iter().any(|&j| subset[j as usize] as usize == i), "point {i} is its own neighbour");
            a.sort_unstable();
            b.sort_unstable();
            differ += (a != b) as usize;
        }
        assert!(differ <= n / 100, "{differ} of {n} neighbour lists differ from the host");
    }

    /// Device k-means++ (no host round trip per seed) puts a seed in every well-separated blob and
    /// never repeats a point while points remain apart from every seed.
    #[test]
    fn seeds_cover_separated_blobs() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (b, per, d) = (5usize, 300usize, 4usize);
        let n = b * per;
        let points = DevicePoints::upload(&client, &blobs(b, per, d, 20.0), n, d);
        let seeds = kmeans_plus_plus_seeds(&client, &points, 12, &mut Rng(3));
        assert_eq!(seeds.len(), 12);
        let covered: std::collections::BTreeSet<usize> = seeds.iter().map(|&s| s as usize / per).collect();
        assert_eq!(covered.len(), b, "seeds {seeds:?} miss a blob");
        let distinct: std::collections::BTreeSet<u32> = seeds.iter().copied().collect();
        assert_eq!(distinct.len(), seeds.len(), "repeated seed in {seeds:?}");
    }

    /// Well-separated blobs come out as at least that many clusters, none mixing two blobs (the
    /// method oversplits by design; the merge tree joins pieces later).
    #[test]
    fn separated_blobs_are_not_mixed() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (b, per, d) = (3usize, 400usize, 6usize);
        let n = b * per;
        let points = DevicePoints::upload(&client, &blobs(b, per, d, 12.0), n, d);
        let opts = BipartiteOptions { subset_stride: 2, init_clusters: 20, ..Default::default() };
        let res = bipartite_clustering(&client, &points, &opts);
        let labels = buffer::download_prefix::<u32>(&client, res.labels.clone(), n);
        assert!(res.n_clusters >= b, "{} clusters for {b} blobs", res.n_clusters);
        assert!(labels.iter().all(|&l| (l as usize) < res.n_clusters));
        for c in 0..res.n_clusters as u32 {
            let blobs_in: std::collections::BTreeSet<usize> = (0..n).filter(|&i| labels[i] == c).map(|i| i / per).collect();
            assert!(blobs_in.len() <= 1, "cluster {c} mixes blobs {blobs_in:?}");
        }
        // Every edge is counted at most once (edges into a right node whose cluster lost all its
        // left nodes in the last round are dropped)
        let total: u64 = res.edges.iter().map(|&e| e as u64).sum();
        assert!(total <= (n * opts.neighbours) as u64 && total * 10 >= (n * opts.neighbours * 9) as u64, "{total} edges");
    }
}
