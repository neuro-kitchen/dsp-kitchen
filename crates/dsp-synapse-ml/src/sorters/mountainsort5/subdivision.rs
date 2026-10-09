//! MountainSort 5's `isosplit6_subdivision_method` (`core/isosplit6_subdivision_method.py`):
//! PCA of the points to `npca_per_subdivision` dimensions, isosplit6 there; when it finds more
//! than one cluster, the cluster medians (in the input space) are split in two by single linkage
//! (the minimum spanning tree of the medians without its longest edge) and each half is
//! subdivided again, its labels numbered after the first half's.

use cubecl::prelude::*;
use cubecl::server::Handle;
use rayon::prelude::*;
use dsp_base::core::buffer;
use dsp_base::linalg::{DeviceRows, TopComponents, TopComponentsOptions};
use dsp_synapse::sorting::{isosplit6, IsosplitOptions};

/// Settings of [`isosplit6_subdivision`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubdivisionOptions {
    pub npca_per_subdivision: usize,
    pub isosplit: IsosplitOptions,
    pub pca: TopComponentsOptions,
}

impl Default for SubdivisionOptions {
    fn default() -> Self {
        Self { npca_per_subdivision: 10, isosplit: IsosplitOptions::default(), pca: TopComponentsOptions::default() }
    }
}

/// Labels `1..=K` of the `l` rows of `x` (row-major `[l, d]`).
///
/// # Panics
///
/// If `x.len() != l · d`.
pub fn isosplit6_subdivision(client: &Client, x: &[f32], l: usize, d: usize, opts: &SubdivisionOptions) -> Vec<u32> {
    isosplit6_subdivision_with_progress(client, x, l, d, opts, &mut |_| {})
}

/// [`isosplit6_subdivision`], calling `done(n)` each time `n` more points have their final label.
pub fn isosplit6_subdivision_with_progress(
    client: &Client,
    x: &[f32],
    l: usize,
    d: usize,
    opts: &SubdivisionOptions,
    done: &mut dyn FnMut(usize),
) -> Vec<u32> {
    assert_eq!(x.len(), l * d, "x must be [l, d]");
    if l == 0 {
        return Vec::new();
    }
    // The points go to the device once; every subset is gathered there
    let device = Points { host: x, device: buffer::upload(client, x), dim: d };
    let all: Vec<usize> = (0..l).collect();
    subdivide(client, &device, &all, opts, done)
}

/// The points on the host (medians) and on the device (PCA).
struct Points<'a> {
    host: &'a [f32],
    device: Handle,
    dim: usize,
}

fn subdivide(client: &Client, points: &Points<'_>, inds: &[usize], opts: &SubdivisionOptions, done: &mut dyn FnMut(usize)) -> Vec<u32> {
    let (x, d) = (points.host, points.dim);
    let l = inds.len();
    if l == 0 {
        return Vec::new();
    }
    if l == 1 || d == 0 {
        done(l);
        return vec![1; l];
    }
    let idx: Vec<u32> = inds.iter().map(|&i| i as u32).collect();
    let rows = DeviceRows { data: &points.device, len: x.len(), dim: d, rows: &idx };
    let pca = TopComponents::fit(client, &rows, opts.npca_per_subdivision, &opts.pca);
    let features: Vec<f64> = pca.transform(client, &rows, opts.pca.batch_elements).into_iter().map(f64::from).collect();
    let labels = isosplit6(&features, l, pca.k, &opts.isosplit);
    let k = labels.iter().copied().max().unwrap_or(0) as usize;
    if k <= 1 {
        done(l);
        return labels;
    }
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &lab) in labels.iter().enumerate() {
        members[lab as usize - 1].push(inds[i]);
    }
    let centroids: Vec<Vec<f64>> = members
        .iter()
        .map(|idx| {
            (0..d)
                .into_par_iter()
                .map_init(Vec::new, |column, j| {
                    column.clear();
                    column.extend(idx.iter().map(|&i| x[i * d + j] as f64));
                    median(column)
                })
                .collect()
        })
        .collect();
    let first = single_linkage_halves(&centroids);
    let (mut i1, mut i2) = (Vec::new(), Vec::new());
    let (mut local1, mut local2) = (Vec::new(), Vec::new());
    for (i, &lab) in labels.iter().enumerate() {
        if first[lab as usize - 1] {
            i1.push(inds[i]);
            local1.push(i);
        } else {
            i2.push(inds[i]);
            local2.push(i);
        }
    }
    let labels1 = subdivide(client, points, &i1, opts, done);
    let labels2 = subdivide(client, points, &i2, opts, done);
    let k1 = labels1.iter().copied().max().unwrap_or(0);
    let mut out = vec![0u32; l];
    for (j, &i) in local1.iter().enumerate() {
        out[i] = labels1[j];
    }
    for (j, &i) in local2.iter().enumerate() {
        out[i] = labels2[j] + k1;
    }
    out
}

/// NumPy's median (the mean of the two middle values for an even count), by selection (`v` is
/// reordered).
fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let (lower, mid, _) = v.select_nth_unstable_by(n / 2, f64::total_cmp);
    let mid = *mid;
    if n % 2 == 1 { mid } else { 0.5 * (lower.iter().copied().fold(f64::NEG_INFINITY, f64::max) + mid) }
}

/// SciPy `cut_tree(linkage(·, 'single'), n_clusters=2)` of the points: removing the longest edge
/// of their minimum spanning tree leaves two groups; `true` marks the group of point 0 (`cut_tree`
/// numbers groups by first appearance).
fn single_linkage_halves(points: &[Vec<f64>]) -> Vec<bool> {
    let k = points.len();
    let dist = |a: usize, b: usize| points[a].iter().zip(&points[b]).map(|(u, v)| (u - v) * (u - v)).sum::<f64>().sqrt();
    // Prim's tree: parent and edge length of every point but 0
    let mut in_tree = vec![false; k];
    let mut best = vec![f64::INFINITY; k];
    let mut parent = vec![0usize; k];
    best[0] = 0.0;
    let mut edges = Vec::with_capacity(k.saturating_sub(1));
    for _ in 0..k {
        let u = (0..k).filter(|&i| !in_tree[i]).min_by(|&a, &b| best[a].total_cmp(&best[b])).expect("a point is left");
        in_tree[u] = true;
        if u != 0 {
            edges.push((best[u], parent[u], u));
        }
        for v in 0..k {
            if !in_tree[v] {
                let dv = dist(u, v);
                if dv < best[v] {
                    best[v] = dv;
                    parent[v] = u;
                }
            }
        }
    }
    // Drop the longest edge (the last single-linkage merge); the first on ties
    let cut = (0..edges.len()).fold(0, |b, i| if edges[i].0 > edges[b].0 { i } else { b });
    let mut group = vec![usize::MAX; k];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &(_, a, b)) in edges.iter().enumerate() {
        if i != cut {
            adj[a].push(b);
            adj[b].push(a);
        }
    }
    let mut stack = vec![0usize];
    group[0] = 0;
    while let Some(u) = stack.pop() {
        for &v in &adj[u] {
            if group[v] == usize::MAX {
                group[v] = 0;
                stack.push(v);
            }
        }
    }
    group.iter().map(|&g| g == 0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;

    #[test]
    fn the_longest_tree_edge_splits_the_medians() {
        let pts = vec![vec![0.0], vec![10.0], vec![1.0], vec![11.0], vec![2.5]];
        assert_eq!(single_linkage_halves(&pts), vec![true, false, true, false, true]);
        assert_eq!(median(&mut [3.0, 1.0, 2.0, 10.0]), 2.5);
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
    }

    /// Five well separated blobs in 30 dimensions are recovered exactly.
    #[test]
    fn subdivision_recovers_separated_clusters() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (l, d, k) = (2000usize, 30usize, 5usize);
        let mut state = 0x2545_f491u64;
        let mut normal = || {
            let mut u = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (u(), u());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        };
        let mut x = vec![0.0f32; l * d];
        let truth: Vec<usize> = (0..l).map(|i| i % k).collect();
        for i in 0..l {
            for j in 0..d {
                let centre = if j == truth[i] * 3 { 12.0 } else { 0.0 };
                x[i * d + j] = (centre + normal()) as f32;
            }
        }
        let labels = isosplit6_subdivision(&client, &x, l, d, &SubdivisionOptions::default());
        assert_eq!(*labels.iter().max().unwrap(), k as u32, "five clusters");
        for c in 0..k {
            let mut seen: Vec<u32> = (0..l).filter(|&i| truth[i] == c).map(|i| labels[i]).collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), 1, "blob {c} split: {seen:?}");
        }
    }
}
