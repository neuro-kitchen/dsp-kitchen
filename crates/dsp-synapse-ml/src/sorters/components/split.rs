//! SpikeInterface's iterative cluster split (`sortingcomponents/clustering/itersplit_tools.py`
//! `split_clusters` + `LocalFeatureClustering`, MIT), shared by SpyKING CIRCUS 2 (HDBSCAN) and
//! Tridesclous 2 (isosplit).
//!
//! Every label is a job, first in, first out (upstream submits them to a pool and reads the
//! results in order). A job (the peaks of one label):
//! 1. channels of the label's peaks; the **intersection** of their neighbourhoods (`split_radius_um`)
//!    and their union; no split when intersection / union < `minimum_overlap_ratio`;
//! 2. each peak's sparse features (`[components, its sparse channels]`) on the intersection
//!    channels; a peak whose sparse channels do not cover them is set aside (label −2); no split
//!    with fewer than `min_size_split` peaks left;
//! 3. the features flattened (component-major), reduced to `n_pca_features` by a truncated SVD
//!    (uncentred) when wider, and clustered (`clusterer`, −1 noise);
//! 4. a split when more than one cluster: the clusters get new labels after the largest so far,
//!    noise and set-aside peaks keep −1 / −2; with `recursive`, every new label (and −2, as
//!    upstream's `setdiff1d(·, [-1])` leaves it) is queued again while the peaks' split count is
//!    below `recursive_depth`.
//!
//! Choice of ours: the truncated SVD is exact (the leading right singular vectors of the small
//! feature matrix, host `f64`); upstream's scikit-learn `TruncatedSVD` is randomized (`n_iter = 5`),
//! equal to it up to that solver's tolerance.

use std::collections::VecDeque;

use dsp_base::linalg::symmetric_eigen_cpu;
use dsp_synapse::features::ChannelNeighbourhoods;

/// Settings of [`split_clusters`] (SpyKING CIRCUS 2's in [`Default`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitOptions {
    pub split_radius_um: f32,
    pub recursive: bool,
    pub recursive_depth: usize,
    pub min_size_split: usize,
    pub n_pca_features: usize,
    pub minimum_overlap_ratio: f64,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self { split_radius_um: 75.0, recursive: true, recursive_depth: 3, min_size_split: 40, n_pca_features: 3, minimum_overlap_ratio: 0.25 }
    }
}

/// Every peak's features on its own sparse channels: `[peaks, components, max_neighbours]`, slot `k`
/// the `k`-th channel of `mask.of(channel)`.
pub struct SparseFeatures<'a> {
    pub data: &'a [f32],
    pub components: usize,
    pub mask: &'a ChannelNeighbourhoods,
}

/// New labels of the peaks (module docs). `labels`: starting labels (e.g. the peak channels; −1
/// is left alone); `clusterer(x, n, d)`: labels of `n` points of `d` features (−1 noise).
pub fn split_clusters(
    labels: &[i64],
    peak_channels: &[u32],
    features: &SparseFeatures<'_>,
    positions: &[[f32; 2]],
    opts: &SplitOptions,
    clusterer: &mut dyn FnMut(&[f32], usize, usize) -> Vec<i32>,
) -> Vec<i64> {
    let m = positions.len();
    let near: Vec<bool> = (0..m * m)
        .map(|e| {
            let (a, b) = (positions[e / m], positions[e % m]);
            (a[0] - b[0]).hypot(a[1] - b[1]) <= opts.split_radius_um
        })
        .collect();
    let mut labels = labels.to_vec();
    let mut split_count = vec![0usize; labels.len()];
    let mut set: Vec<i64> = labels.iter().copied().filter(|&l| l != -1).collect();
    set.sort_unstable();
    set.dedup();
    let Some(&max) = set.last() else { return labels };
    let mut current_max = max + 1;
    let members = |labels: &[i64], l: i64| -> Vec<usize> { (0..labels.len()).filter(|&i| labels[i] == l).collect() };
    let mut by_label: std::collections::BTreeMap<i64, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if l != -1 {
            by_label.entry(l).or_default().push(i);
        }
    }
    let mut jobs: VecDeque<Vec<usize>> = by_label.into_values().collect();
    while let Some(peaks) = jobs.pop_front() {
        let Some(local) = split_one(&peaks, peak_channels, features, &near, m, opts, clusterer) else { continue };
        let kmax = local.iter().copied().filter(|&l| l >= 0).max().unwrap_or(0) as i64;
        for (j, &i) in peaks.iter().enumerate() {
            labels[i] = if local[j] >= 0 { local[j] as i64 + current_max } else { local[j] as i64 };
            split_count[i] += 1;
        }
        current_max += kmax + 1;
        if opts.recursive {
            let level = peaks.iter().map(|&i| split_count[i]).max().unwrap_or(0);
            if level < opts.recursive_depth {
                let mut new: Vec<i64> = peaks.iter().map(|&i| labels[i]).filter(|&l| l != -1).collect();
                new.sort_unstable();
                new.dedup();
                for l in new {
                    // A new label holds only this job's peaks; −2 is collected everywhere, as upstream
                    let idx = if l == -2 { members(&labels, l) } else { peaks.iter().copied().filter(|&i| labels[i] == l).collect() };
                    if !idx.is_empty() {
                        jobs.push_back(idx);
                    }
                }
            }
        }
    }
    labels
}

/// One job: local labels of `peaks` when they split (module docs), else `None`.
fn split_one(
    peaks: &[usize],
    peak_channels: &[u32],
    f: &SparseFeatures<'_>,
    near: &[bool],
    m: usize,
    opts: &SplitOptions,
    clusterer: &mut dyn FnMut(&[f32], usize, usize) -> Vec<i32>,
) -> Option<Vec<i32>> {
    let mut chans: Vec<usize> = peaks.iter().map(|&i| peak_channels[i] as usize).collect();
    chans.sort_unstable();
    chans.dedup();
    let inter: Vec<usize> = (0..m).filter(|&c| chans.iter().all(|&a| near[a * m + c])).collect();
    let union = (0..m).filter(|&c| chans.iter().any(|&a| near[a * m + c])).count();
    if union == 0 || (inter.len() as f64 / union as f64) < opts.minimum_overlap_ratio {
        return None;
    }
    let (comps, nb) = (f.components, f.mask.max_neighbours);
    let k = inter.len();
    let mut local = vec![0i32; peaks.len()];
    let mut x: Vec<f32> = Vec::new();
    let mut kept = Vec::new();
    // The slot of each intersection channel in each peak channel's sparse row (None: missing)
    for (j, &i) in peaks.iter().enumerate() {
        let ch = peak_channels[i] as usize;
        let row: Vec<usize> = f.mask.of(ch).collect();
        let slots: Option<Vec<usize>> = inter.iter().map(|c| row.iter().position(|r| r == c)).collect();
        match slots {
            None => local[j] = -2,
            Some(slots) => {
                for c in 0..comps {
                    for &s in &slots {
                        x.push(f.data[(i * comps + c) * nb + s]);
                    }
                }
                kept.push(j);
            }
        }
    }
    if kept.len() < opts.min_size_split {
        return None;
    }
    let (n, d) = (kept.len(), comps * k);
    let (x, d) = if d > opts.n_pca_features { (truncated_svd(&x, n, d, opts.n_pca_features), opts.n_pca_features) } else { (x, d) };
    let labels = clusterer(&x, n, d);
    let mut distinct: Vec<i32> = labels.iter().copied().filter(|&l| l != -1).collect();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() <= 1 {
        return None;
    }
    for (j, &l) in kept.iter().zip(&labels) {
        local[*j] = l;
    }
    Some(local)
}

/// Projections of the rows of `x` (`[n, d]`) on its `k` leading right singular vectors (uncentred,
/// as `TruncatedSVD`): `x · V`, `[n, k]`.
pub fn truncated_svd(x: &[f32], n: usize, d: usize, k: usize) -> Vec<f32> {
    let mut g = vec![0.0f64; d * d];
    for row in x.chunks_exact(d) {
        for a in 0..d {
            let ra = row[a] as f64;
            for b in a..d {
                g[a * d + b] += ra * row[b] as f64;
            }
        }
    }
    for a in 0..d {
        for b in 0..a {
            g[a * d + b] = g[b * d + a];
        }
    }
    let eig = symmetric_eigen_cpu(&g, d);
    let k = k.min(d);
    let mut out = Vec::with_capacity(n * k);
    for row in x.chunks_exact(d) {
        for c in 0..k {
            out.push((0..d).map(|a| row[a] as f64 * eig.vectors[a * d + c]).sum::<f64>() as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two waveform shapes on one channel split; a third level (depth 2) re-splits nothing.
    #[test]
    fn splits_two_shapes_and_sets_aside_uncovered_peaks() {
        let pos: Vec<[f32; 2]> = (0..3).map(|c| [0.0, 20.0 * c as f32]).collect();
        let mask = ChannelNeighbourhoods::within_radius(&pos, 25.0);
        let comps = 2;
        let nb = mask.max_neighbours;
        // 100 peaks on channel 1 (sparse channels 0, 1, 2), half of each shape; 3 on channel 0
        let n = 103;
        let mut data = vec![0.0f32; n * comps * nb];
        let mut channels = vec![1u32; n];
        let mut state = 1u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        for i in 0..100 {
            let a = if i % 2 == 0 { 5.0 } else { -5.0 };
            for c in 0..comps {
                for s in 0..3 {
                    data[(i * comps + c) * nb + s] = a * (c as f32 + 1.0) + rnd();
                }
            }
        }
        for i in 100..103 {
            channels[i] = 0;
        }
        let f = SparseFeatures { data: &data, components: comps, mask: &mask };
        let start: Vec<i64> = channels.iter().map(|&c| c as i64).collect();
        let opts = SplitOptions { split_radius_um: 25.0, min_size_split: 10, recursive_depth: 2, ..Default::default() };
        // A clusterer splitting on the sign of the first feature
        let mut by_sign = |x: &[f32], n: usize, d: usize| -> Vec<i32> { (0..n).map(|i| (x[i * d] > 0.0) as i32).collect() };
        let labels = split_clusters(&start, &channels, &f, &pos, &opts, &mut by_sign);
        let a: Vec<i64> = (0..100).step_by(2).map(|i| labels[i]).collect();
        let b: Vec<i64> = (1..100).step_by(2).map(|i| labels[i]).collect();
        assert!(a.iter().all(|&l| l == a[0]) && b.iter().all(|&l| l == b[0]) && a[0] != b[0], "{labels:?}");
        assert!(a[0] >= 2 && b[0] >= 2, "new labels follow the largest");
    }

    #[test]
    fn truncated_svd_recovers_a_rank_one_direction() {
        let d = 4;
        let x: Vec<f32> = (0..50).flat_map(|i| { let t = i as f32 - 25.0; [t, 2.0 * t, 0.0, -t] }).collect();
        let p = truncated_svd(&x, 50, d, 1);
        let norm = (1.0f32 + 4.0 + 1.0).sqrt();
        for i in 0..50 {
            let t = i as f32 - 25.0;
            assert!((p[i].abs() - (t * norm).abs()).abs() < 1e-3, "{} vs {}", p[i], t * norm);
        }
    }
}
