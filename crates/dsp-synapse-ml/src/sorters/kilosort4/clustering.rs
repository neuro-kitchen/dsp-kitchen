//! Kilosort4's first clustering of the universal-template spikes into units, written from the
//! paper (Pachitariu et al. 2024, Methods: *Template learning*, *Graph-based clustering*,
//! *Hierarchical merging tree*, *Split/merge criteria*, *Bimodality of regression axis*).
//!
//! 1. **Sections.** Spikes are binned by vertical position (`section_um`, 40 µm), per shank. A
//!    section's spikes are embedded on the union of their centres' channels: `[slots · n_pcs]`
//!    features, 0 on channels a spike's centre does not cover. Built on the device from the compact
//!    `[n, nearest_chans, n_pcs]` features, so only those cross the bus.
//! 2. **Graph clustering** of each section of at least `min_section_spikes` spikes
//!    ([`dsp_synapse::sorting::bipartite_clustering`]): oversplit on purpose.
//! 3. **Merging tree** (host: at most `init_clusters` leaves). From the edge counts `K` between
//!    clusters and their degree sums `k`, the pair with the highest `γ̂ = 2m·K_ij / (k_i·k_j)` (the
//!    resolution at which merging them stops lowering modularity) is merged, repeatedly: a tree
//!    whose merges come in decreasing `γ̂`.
//! 4. **Decisions**, from the top: a node is split when `γ̂ < modularity_split`, or when its two
//!    halves are bimodal along their weighted regression axis (score ≥ `bimodality_split`); the
//!    halves of a split node are decided in turn, an unsplit node is one unit. (Refractory cross-
//!    correlograms, the paper's other criterion, are not used in this first clustering.)
//! 5. **Templates:** each unit's mean features, on all channels (`[units, channels, n_pcs]`).
//!
//! **Bimodality** (on the device): the axis `u` and bias `b` minimizing `Σ w_y (uᵀx + b − y)²` with
//! `y = ∓1` per half and `w₋ = n₊/(n₋+n₊)`, `w₊ = n₋/(n₋+n₊)` (normal equations: `Σ w x xᵀ`,
//! `Σ w y x`, `Σ w x` through `matmul`; `Σ w` and `Σ w y = 0` follow from the sizes; solved on the
//! host in `f64` with a small ridge); projections binned in 400 bins over `[−2, 2]`, smoothed by
//! a Gaussian of 4 bins; trough `x_min` in bins 175–225, peaks `x₁`, `x₂` on either side; score
//! `1 − max(x_min/x₁, x_min/x₂)`.
//!
//! Choices of ours where the paper is silent: the bias `b` (the paper's axis passes through the
//! origin, but spike features are not centred: without it a half sitting at the origin projects
//! into the trough), the ridge (`RIDGE` × mean diagonal), the smoothing
//! kernel truncated at 4 σ with mirrored edges, a side without any projection mass scores 0 (not
//! bimodal), merge-tree ties to the smallest pair.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::linalg::{matmul, MatrixView};
use dsp_core::compute::LaunchGeometry;
use dsp_io::neuro::probe::SensorLayout;
use dsp_synapse::sorting::kernels::bipartite::{relabel_kernel, NO_LABEL};
use dsp_synapse::sorting::{bipartite_clustering, cluster_sums, BipartiteOptions, DevicePoints};

use super::detect::{TemplateCentres, UniversalSpike};
use super::kernels::cluster::{embed_section_kernel, projection_histogram_kernel, weigh_node_kernel, SIDE_NEG, SIDE_NONE, SIDE_POS};

/// Bins of the regression-axis histogram, its range, smoothing (bins) and trough search range.
const HIST_BINS: usize = 400;
const HIST_RANGE: f32 = 2.0;
const HIST_SMOOTH_SD: f64 = 4.0;
const TROUGH_BINS: std::ops::Range<usize> = 175..225;
/// Ridge of the regression, relative to the mean diagonal of `Σ w x xᵀ`.
const RIDGE: f64 = 1e-6;

/// Settings of [`cluster_spikes`] (paper / upstream defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusteringOptions {
    /// Height of a probe section (µm).
    pub section_um: f32,
    /// Sections with fewer spikes are one unit.
    pub min_section_spikes: usize,
    pub graph: BipartiteOptions,
    /// Nodes of the merging tree below this `γ̂` are always split.
    pub modularity_split: f64,
    /// Nodes whose halves score at least this bimodality are split.
    pub bimodality_split: f64,
}

impl Default for ClusteringOptions {
    fn default() -> Self {
        Self { section_um: 40.0, min_section_spikes: 1000, graph: BipartiteOptions::default(), modularity_split: 0.2, bimodality_split: 0.6 }
    }
}

/// Units of the spikes (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct SpikeClusters {
    /// Unit of every spike (in the order the spikes were given).
    pub labels: Vec<u32>,
    pub n_units: usize,
    /// `[n_units, channels, n_pcs]` mean features.
    pub templates: Vec<f32>,
    pub channels: usize,
    pub n_pcs: usize,
    /// Sections clustered (non-empty).
    pub sections: usize,
}

/// One merge of the tree: children (`< leaves`: a leaf, else node `child − leaves`) and its `γ̂`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Merge {
    pub a: usize,
    pub b: usize,
    pub gamma: f64,
}

/// Merging tree of `nc` clusters from their `[nc, nc]` edge counts (left cluster → right cluster):
/// `nc − 1` merges in order (see the module docs).
pub fn merge_tree(edges: &[u32], nc: usize) -> Vec<Merge> {
    assert_eq!(edges.len(), nc * nc);
    let total: f64 = edges.iter().map(|&e| e as f64).sum();
    // Between-cluster edges (both directions) and degree sums (left and right ends), per active node
    let mut k_between = vec![vec![0.0f64; nc]; nc];
    let mut degree = vec![0.0f64; nc];
    for i in 0..nc {
        for j in 0..nc {
            let e = edges[i * nc + j] as f64;
            degree[i] += e;
            degree[j] += e;
            if i != j {
                k_between[i][j] += e;
                k_between[j][i] += e;
            }
        }
    }
    let mut id: Vec<usize> = (0..nc).collect();
    let mut active = vec![true; nc];
    let mut merges = Vec::with_capacity(nc.saturating_sub(1));
    let gamma = |kb: f64, ki: f64, kj: f64| if ki > 0.0 && kj > 0.0 { 2.0 * total * kb / (ki * kj) } else { 0.0 };
    for step in 0..nc.saturating_sub(1) {
        let mut best: Option<(f64, usize, usize)> = None;
        for i in (0..nc).filter(|&i| active[i]) {
            for j in (i + 1..nc).filter(|&j| active[j]) {
                let g = gamma(k_between[i][j], degree[i], degree[j]);
                if best.is_none_or(|(bg, _, _)| g > bg) {
                    best = Some((g, i, j));
                }
            }
        }
        let (g, i, j) = best.expect("two active clusters");
        merges.push(Merge { a: id[i], b: id[j], gamma: g });
        // j joins i
        for l in 0..nc {
            if l != i && l != j {
                k_between[i][l] += k_between[j][l];
                k_between[l][i] = k_between[i][l];
            }
        }
        degree[i] += degree[j];
        active[j] = false;
        id[i] = nc + step;
    }
    merges
}

/// Gaussian smoothing with standard deviation `sd` (bins), kernel truncated at 4 `sd`, mirrored edges.
fn smooth(hist: &[u32], sd: f64) -> Vec<f64> {
    let radius = (4.0 * sd).ceil() as isize;
    let w: Vec<f64> = (-radius..=radius).map(|o| (-0.5 * (o as f64 / sd).powi(2)).exp()).collect();
    let norm: f64 = w.iter().sum();
    let n = hist.len() as isize;
    let at = |i: isize| {
        // Mirror: −1 → 0, n → n − 1 (half-sample symmetric)
        let j = if i < 0 { -i - 1 } else if i >= n { 2 * n - i - 1 } else { i };
        hist[j.clamp(0, n - 1) as usize] as f64
    };
    (0..n).map(|i| (-radius..=radius).zip(&w).map(|(o, &wo)| wo * at(i + o)).sum::<f64>() / norm).collect()
}

/// The paper's bimodality score of a 400-bin projection histogram (see the module docs).
pub fn bimodality_score(hist: &[u32]) -> f64 {
    assert_eq!(hist.len(), HIST_BINS);
    let h = smooth(hist, HIST_SMOOTH_SD);
    let imin = TROUGH_BINS.fold(TROUGH_BINS.start, |b, i| if h[i] < h[b] { i } else { b });
    let xmin = h[imin];
    let x1 = h[..imin].iter().copied().fold(0.0, f64::max);
    let x2 = h[imin..].iter().copied().fold(0.0, f64::max);
    if x1 <= 0.0 || x2 <= 0.0 {
        return 0.0;
    }
    1.0 - (xmin / x1).max(xmin / x2)
}

/// `(G + λI) u = h` for a symmetric positive semi-definite `d × d` `G` (Cholesky, `f64`).
fn solve_ridge(mut g: Vec<f64>, h: &[f64], d: usize) -> Vec<f64> {
    let mean_diag = (0..d).map(|i| g[i * d + i]).sum::<f64>() / d.max(1) as f64;
    let ridge = RIDGE * mean_diag.max(f64::MIN_POSITIVE);
    for i in 0..d {
        g[i * d + i] += ridge;
    }
    // In-place Cholesky: lower triangle holds L
    for j in 0..d {
        let mut diag = g[j * d + j];
        for k in 0..j {
            diag -= g[j * d + k] * g[j * d + k];
        }
        let diag = diag.max(ridge).sqrt();
        g[j * d + j] = diag;
        for i in j + 1..d {
            let mut v = g[i * d + j];
            for k in 0..j {
                v -= g[i * d + k] * g[j * d + k];
            }
            g[i * d + j] = v / diag;
        }
    }
    let mut z = h.to_vec();
    for i in 0..d {
        for k in 0..i {
            z[i] -= g[i * d + k] * z[k];
        }
        z[i] /= g[i * d + i];
    }
    for i in (0..d).rev() {
        for k in i + 1..d {
            z[i] -= g[k * d + i] * z[k];
        }
        z[i] /= g[i * d + i];
    }
    z
}

/// A clustered section on the device: its points, leaf labels and leaf sizes.
struct Section<'a> {
    client: &'a Client,
    points: &'a DevicePoints,
    labels: &'a Handle,
    sizes: Vec<usize>,
    /// Scratch of [`Section::bimodality`]: weighted points, weighted targets, `√w`, Gram, `Σ w y x`,
    /// `Σ w x`.
    xs: Handle,
    v: Handle,
    s: Handle,
    gram: Handle,
    rhs: Handle,
    mean: Handle,
}

impl<'a> Section<'a> {
    fn new(client: &'a Client, points: &'a DevicePoints, labels: &'a Handle, sizes: Vec<usize>) -> Self {
        let (n, d) = (points.n, points.d);
        Self {
            client,
            points,
            labels,
            sizes,
            xs: buffer::empty::<f32>(client, d * n),
            v: buffer::empty::<f32>(client, n),
            s: buffer::empty::<f32>(client, n),
            gram: buffer::empty::<f32>(client, d * d),
            rhs: buffer::empty::<f32>(client, d),
            mean: buffer::empty::<f32>(client, d),
        }
    }

    /// Bimodality score of the leaves `neg` against the leaves `pos` (module docs). Crosses the bus:
    /// `side` and `u` up (`nc` + `d` values), `Σ w x xᵀ`, `Σ w y x`, `Σ w x` and the histogram down.
    fn bimodality(&self, neg: &[usize], pos: &[usize]) -> f64 {
        let (client, n, d, nc) = (self.client, self.points.n, self.points.d, self.sizes.len());
        let (n_neg, n_pos) = (neg.iter().map(|&l| self.sizes[l]).sum::<usize>() as f64, pos.iter().map(|&l| self.sizes[l]).sum::<usize>() as f64);
        let mut side = vec![SIDE_NONE; nc];
        neg.iter().for_each(|&l| side[l] = SIDE_NEG);
        pos.iter().for_each(|&l| side[l] = SIDE_POS);
        let side = buffer::upload(client, &side);
        // Balanced weights: each half weighs by the other half's share
        let (w_neg, w_pos) = (n_pos / (n_neg + n_pos), n_neg / (n_neg + n_pos));
        let per_point = LaunchGeometry::elementwise(client, n);
        // SAFETY: `points`, `xs` hold `d · n`, `labels`, `v` `n`, `side` `nc` values
        unsafe {
            weigh_node_kernel::launch::<f32>(
                client,
                per_point.cube_count.clone(),
                per_point.cube_dim.clone(),
                BufferArg::from_raw_parts(self.points.handle.clone(), d * n),
                BufferArg::from_raw_parts(self.labels.clone(), n),
                BufferArg::from_raw_parts(side.clone(), nc),
                BufferArg::from_raw_parts(self.xs.clone(), d * n),
                BufferArg::from_raw_parts(self.v.clone(), n),
                BufferArg::from_raw_parts(self.s.clone(), n),
                n as u32,
                d as u32,
                nc as u32,
                w_neg.sqrt() as f32,
                w_pos.sqrt() as f32,
            );
        }
        let xs = MatrixView::row_major(&self.xs, d * n, d, n);
        matmul::<f32>(client, &xs, &xs.transposed(), &self.gram, d * d);
        matmul::<f32>(client, &xs, &MatrixView::row_major(&self.v, n, n, 1), &self.rhs, d);
        matmul::<f32>(client, &xs, &MatrixView::row_major(&self.s, n, n, 1), &self.mean, d);
        let down = |h: &Handle, len: usize| -> Vec<f64> { buffer::download_prefix::<f32>(client, h.clone(), len).into_iter().map(f64::from).collect() };
        let (g, h, sx) = (down(&self.gram, d * d), down(&self.rhs, d), down(&self.mean, d));
        // Augmented system over (u, b): [[Σwxxᵀ, Σwx], [Σwxᵀ, Σw]] (u, b) = (Σwyx, Σwy), Σwy = 0
        let e = d + 1;
        let mut ga = vec![0.0f64; e * e];
        for i in 0..d {
            ga[i * e..i * e + d].copy_from_slice(&g[i * d..(i + 1) * d]);
            ga[i * e + d] = sx[i];
            ga[d * e + i] = sx[i];
        }
        ga[d * e + d] = w_neg * n_neg + w_pos * n_pos;
        let mut ha = h;
        ha.push(0.0);
        let ub = solve_ridge(ga, &ha, e);
        let u: Vec<f32> = ub[..d].iter().map(|&v| v as f32).collect();
        let bias = ub[d] as f32;
        let hist = buffer::zeros::<u32>(client, HIST_BINS);
        // SAFETY: as above; `u` holds `d`, `hist` `HIST_BINS` values
        unsafe {
            projection_histogram_kernel::launch::<f32>(
                client,
                per_point.cube_count,
                per_point.cube_dim,
                BufferArg::from_raw_parts(self.points.handle.clone(), d * n),
                BufferArg::from_raw_parts(self.labels.clone(), n),
                BufferArg::from_raw_parts(side, nc),
                BufferArg::from_raw_parts(buffer::upload(client, &u), d),
                BufferArg::from_raw_parts(hist.clone(), HIST_BINS),
                n as u32,
                d as u32,
                nc as u32,
                HIST_BINS as u32,
                bias,
                -HIST_RANGE,
                HIST_RANGE,
            );
        }
        bimodality_score(&buffer::download_prefix::<u32>(client, hist, HIST_BINS))
    }
}

/// Leaves under tree node `node` (`< nc`: a leaf).
fn leaves(merges: &[Merge], nc: usize, node: usize, out: &mut Vec<usize>) {
    if node < nc {
        out.push(node);
    } else {
        let m = merges[node - nc];
        leaves(merges, nc, m.a, out);
        leaves(merges, nc, m.b, out);
    }
}

/// Top-down decisions (module docs): the units, as groups of leaves. `split(neg, pos)` scores a
/// node's bimodality.
fn decide(merges: &[Merge], nc: usize, opts: &ClusteringOptions, split: &mut dyn FnMut(&[usize], &[usize]) -> f64) -> Vec<Vec<usize>> {
    let mut units = Vec::new();
    if nc == 0 {
        return units;
    }
    let mut stack = vec![if merges.is_empty() { 0 } else { nc + merges.len() - 1 }];
    while let Some(node) = stack.pop() {
        if node < nc {
            units.push(vec![node]);
            continue;
        }
        let m = merges[node - nc];
        let (mut a, mut b) = (Vec::new(), Vec::new());
        leaves(merges, nc, m.a, &mut a);
        leaves(merges, nc, m.b, &mut b);
        if m.gamma < opts.modularity_split || split(&a, &b) >= opts.bimodality_split {
            stack.push(m.b);
            stack.push(m.a);
        } else {
            a.extend(b);
            a.sort_unstable();
            units.push(a);
        }
    }
    units
}

/// Clusters `spikes` (their `features`, `[centres.n_chans, n_pcs]` each) into units, section by
/// section (module docs). `probe` gives each channel's shank; `channels` is the recording's.
pub fn cluster_spikes(
    client: &Client,
    spikes: &[UniversalSpike],
    centres: &TemplateCentres,
    probe: &SensorLayout,
    channels: usize,
    n_pcs: usize,
    opts: &ClusteringOptions,
    progress: &mut dyn FnMut(u64, u64),
) -> SpikeClusters {
    let (n_chans, n_centres) = (centres.n_chans, centres.n_centres());
    let shank_of = |ch: usize| probe.contacts.get(ch).map_or(0, |c| c.shank_id);
    let mut shank_y0: std::collections::BTreeMap<usize, f32> = std::collections::BTreeMap::new();
    for (ch, c) in probe.contacts.iter().enumerate() {
        let y0 = shank_y0.entry(shank_of(ch)).or_insert(f32::INFINITY);
        *y0 = y0.min(c.position.y_um);
    }
    // Sections: (shank, vertical bin) of each spike, in spike order within a section
    let mut sections: std::collections::BTreeMap<(usize, i64), Vec<usize>> = std::collections::BTreeMap::new();
    for (i, s) in spikes.iter().enumerate() {
        let shank = shank_of(centres.ic[s.centre] as usize);
        let bin = ((s.y_um - shank_y0.get(&shank).copied().unwrap_or(0.0)) / opts.section_um).floor() as i64;
        sections.entry((shank, bin)).or_default().push(i);
    }

    let mut labels = vec![0u32; spikes.len()];
    let mut templates = Vec::new();
    let mut n_units = 0usize;
    let total = sections.len() as u64;
    progress(0, total);
    for (done, members) in sections.values().enumerate() {
        let n = members.len();
        // Channel slots of the section: the union of its spikes' centre channels
        let mut slot_of = std::collections::BTreeMap::new();
        for &i in members {
            for c in 0..n_chans {
                slot_of.entry(centres.ic[c * n_centres + spikes[i].centre]).or_insert(0usize);
            }
        }
        let section_channels: Vec<u32> = slot_of.keys().copied().collect();
        for (slot, v) in slot_of.values_mut().enumerate() {
            *v = slot;
        }
        let d = section_channels.len() * n_pcs;
        let mut feat = Vec::with_capacity(n * n_chans * n_pcs);
        let mut slots = Vec::with_capacity(n * n_chans);
        for &i in members {
            assert_eq!(spikes[i].features.len(), n_chans * n_pcs, "spike features: one row of n_pcs per centre channel");
            feat.extend_from_slice(&spikes[i].features);
            slots.extend((0..n_chans).map(|c| slot_of[&centres.ic[c * n_centres + spikes[i].centre]] as u32));
        }
        let x = buffer::zeros::<f32>(client, d * n);
        let geom = LaunchGeometry::elementwise(client, n * n_chans * n_pcs);
        // SAFETY: `feat` holds `n · n_chans · n_pcs`, `slots` `n · n_chans` (all < slots), `x` `d · n`
        unsafe {
            embed_section_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(buffer::upload(client, &feat), feat.len()),
                BufferArg::from_raw_parts(buffer::upload(client, &slots), slots.len()),
                BufferArg::from_raw_parts(x.clone(), d * n),
                n as u32,
                n_chans as u32,
                n_pcs as u32,
            );
        }
        let points = DevicePoints { handle: x, n, d };

        // Units of the section, as labels on the device
        let (unit_labels, section_units) = if n < opts.min_section_spikes.max(2) {
            (buffer::zeros::<u32>(client, n), 1)
        } else {
            let graph = bipartite_clustering(client, &points, &opts.graph);
            let nc = graph.n_clusters;
            let sizes = graph.sizes.clone();
            let tree = merge_tree(&graph.edges, nc);
            let section = Section::new(client, &points, &graph.labels, sizes);
            let groups = decide(&tree, nc, opts, &mut |a, b| section.bimodality(a, b));
            let mut map = vec![NO_LABEL; nc];
            for (u, group) in groups.iter().enumerate() {
                group.iter().for_each(|&leaf| map[leaf] = u as u32);
            }
            let per_point = LaunchGeometry::elementwise(client, n);
            // SAFETY: `labels` holds `n`, `map` `nc` values
            unsafe {
                relabel_kernel::launch(
                    client,
                    per_point.cube_count,
                    per_point.cube_dim,
                    BufferArg::from_raw_parts(graph.labels.clone(), n),
                    BufferArg::from_raw_parts(buffer::upload(client, &map), nc),
                    n as u32,
                    nc as u32,
                );
            }
            (graph.labels, groups.len())
        };

        // Mean features of each unit, back on all channels
        let (sums, counts) = cluster_sums(client, &points, &unit_labels, section_units);
        let local = buffer::download_prefix::<u32>(client, unit_labels, n);
        for (&i, &u) in members.iter().zip(&local) {
            labels[i] = (n_units + u as usize) as u32;
        }
        for u in 0..section_units {
            let mut t = vec![0.0f32; channels * n_pcs];
            for (slot, &ch) in section_channels.iter().enumerate() {
                for p in 0..n_pcs {
                    t[ch as usize * n_pcs + p] = (sums[u * d + slot * n_pcs + p] / counts[u].max(1) as f64) as f32;
                }
            }
            templates.extend(t);
        }
        n_units += section_units;
        progress(done as u64 + 1, total);
    }
    SpikeClusters { labels, n_units, templates, channels, n_pcs, sections: sections.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two groups joined by many edges merge first; the tree has `nc − 1` merges with `γ̂` not
    /// increasing (the paper's monotonic property).
    #[test]
    fn merge_tree_joins_the_most_connected_first() {
        // Clusters 0 and 1 share many edges, 2 is linked weakly to both
        let nc = 3;
        let edges = [50, 40, 2, 40, 50, 2, 2, 2, 60];
        let tree = merge_tree(&edges, nc);
        assert_eq!(tree.len(), 2);
        assert_eq!((tree[0].a, tree[0].b), (0, 1));
        assert_eq!((tree[1].a, tree[1].b), (3, 2), "the merged node then takes cluster 2");
        assert!(tree[0].gamma >= tree[1].gamma);
    }

    /// Two separated bumps score near 1, one bump near 0.
    #[test]
    fn bimodality_scores_two_bumps_high_and_one_low() {
        let bump = |centre: f64, i: usize| (1000.0 * (-0.5 * ((i as f64 - centre) / 15.0).powi(2)).exp()) as u32;
        let two: Vec<u32> = (0..HIST_BINS).map(|i| bump(100.0, i) + bump(300.0, i)).collect();
        let one: Vec<u32> = (0..HIST_BINS).map(|i| bump(200.0, i)).collect();
        assert!(bimodality_score(&two) > 0.9, "{}", bimodality_score(&two));
        assert!(bimodality_score(&one) < 0.1, "{}", bimodality_score(&one));
    }

    #[test]
    fn ridge_solve_matches_a_known_system() {
        // [[4, 1], [1, 3]] u = [1, 2] → u = [1/11, 7/11]
        let u = solve_ridge(vec![4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2);
        assert!((u[0] - 1.0 / 11.0).abs() < 1e-5 && (u[1] - 7.0 / 11.0).abs() < 1e-5, "{u:?}");
    }

    /// Deterministic points `[n, d]`: `groups` blobs (sums of uniforms) `spread` apart on axis 0.
    fn blob_points(groups: usize, per: usize, d: usize, spread: f32) -> Vec<f32> {
        let mut state = 0x2545_f491u32;
        let mut uniform = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let mut x = Vec::with_capacity(groups * per * d);
        for g in 0..groups {
            for _ in 0..per {
                for f in 0..d {
                    let noise: f32 = (0..4).map(|_| uniform()).sum();
                    x.push(noise + if f == 0 { g as f32 * spread } else { 0.0 });
                }
            }
        }
        x
    }

    /// On the device: two separated blobs as the halves score bimodal, two halves of one blob do
    /// not; the weighted Gram equals the host's.
    #[test]
    fn device_bimodality_separates_blobs_only() {
        let Ok(target) = dsp_core::compute::ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (per, d) = (600usize, 5usize);
        let n = 2 * per;
        let x = blob_points(2, per, d, 6.0);
        let points = DevicePoints::upload(&client, &x, n, d);
        // Leaves: 0 = first blob, 1 = second; then 2 / 3 = alternate points of the first blob
        let two: Vec<u32> = (0..n).map(|i| (i / per) as u32).collect();
        let labels = buffer::upload(&client, &two);
        let section = Section::new(&client, &points, &labels, vec![per, per]);
        let score = section.bimodality(&[0], &[1]);
        assert!(score > 0.8, "separated blobs score {score}");
        // Weighted Gram of the last call (equal halves: w = 1/2 each) against the host
        let gram = buffer::download_prefix::<f32>(&client, section.gram.clone(), d * d);
        for (a, b) in [(0usize, 0usize), (0, 3), (2, 4)] {
            let want: f64 = (0..n).map(|i| 0.5 * x[i * d + a] as f64 * x[i * d + b] as f64).sum();
            assert!((gram[a * d + b] as f64 - want).abs() <= 1e-3 * want.abs().max(1.0), "G[{a},{b}] {} vs {want}", gram[a * d + b]);
        }
        let halves: Vec<u32> = (0..per).map(|i| 2 + (i % 2) as u32).collect();
        let labels = buffer::upload(&client, &halves);
        let one = DevicePoints::upload(&client, &x[..per * d], per, d);
        let section = Section::new(&client, &one, &labels, vec![0, 0, per / 2, per / 2]);
        let score = section.bimodality(&[2], &[3]);
        assert!(score < 0.5, "one blob split in two scores {score}");
    }

    /// Splits follow the scores: a weakly connected root splits, a bimodal child splits, a
    /// unimodal one stays one unit.
    #[test]
    fn decisions_walk_the_tree_from_the_top() {
        // Leaves 0..4; merges: (0,1)→4, (2,3)→5, (4,5)→6 with a low γ̂ at the root
        let tree = [Merge { a: 0, b: 1, gamma: 5.0 }, Merge { a: 2, b: 3, gamma: 4.0 }, Merge { a: 4, b: 5, gamma: 0.1 }];
        let opts = ClusteringOptions::default();
        // Bimodal only for (0 | 1)
        let mut units = decide(&tree, 4, &opts, &mut |a, b| if a == [0] && b == [1] { 0.9 } else { 0.1 });
        units.sort();
        assert_eq!(units, vec![vec![0], vec![1], vec![2, 3]]);
    }
}
