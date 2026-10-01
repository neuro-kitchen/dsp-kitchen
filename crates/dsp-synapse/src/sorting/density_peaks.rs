//! Rodriguez-Laio Density Peaks Clustering (`density_peaks.rs`).
//!
//! Unsupervised non-parametric clustering over `[N, D]` spatial-PCA or deep latent
//! embeddings (`(x, y, PC1, PC2)`):
//! 1. Local Gaussian density: $\rho_i = \sum_{j \neq i} \exp(-(d_{ij} / d_c)^2)$
//! 2. Minimum distance to higher density: $\delta_i = \min_{j : \rho_j > \rho_i} d_{ij}$
//! 3. Decision score: $\gamma_i = \rho_i \cdot \delta_i$

use serde::{Deserialize, Serialize};

/// Result of Rodriguez-Laio Density Peaks clustering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DensityPeaksResult {
    /// Assigned cluster ID (`0..num_clusters`) for each of the $N$ spikes.
    pub labels: Vec<usize>,
    /// Indices of the $K$ cluster center exemplars in `0..N`.
    pub cluster_centers: Vec<usize>,
    /// Local density $\rho_i$ for each point.
    pub densities: Vec<f32>,
    /// Distance $\delta_i$ to nearest higher-density neighbor for each point.
    pub deltas: Vec<f32>,
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Runs Rodriguez-Laio Density Peaks clustering on a flat `[num_points, dim]` feature matrix.
///
/// Exact, O(N²·D) time and O(N) extra memory (distances are recomputed, never stored). For large
/// N use [`cluster_density_peaks_capped`].
pub fn cluster_density_peaks(
    features: &[f32],
    num_points: usize,
    dim: usize,
    cutoff_distance: f32,
    num_clusters: usize,
) -> DensityPeaksResult {
    if num_points == 0 || dim == 0 || num_clusters == 0 {
        return DensityPeaksResult {
            labels: Vec::new(),
            cluster_centers: Vec::new(),
            densities: Vec::new(),
            deltas: Vec::new(),
        };
    }
    assert_eq!(features.len(), num_points * dim);
    let row = |i: usize| &features[i * dim..(i + 1) * dim];

    let dc_sq = (cutoff_distance.max(1e-4)) * (cutoff_distance.max(1e-4));

    // 1. Gaussian local densities rho_i
    let mut densities = vec![0.0f32; num_points];
    for i in 0..num_points {
        for j in (i + 1)..num_points {
            let kernel = (-dist2(row(i), row(j)) / dc_sq).exp();
            densities[i] += kernel;
            densities[j] += kernel;
        }
    }

    // 2. Sort point indices in descending order of density rho (ties by index)
    let mut order: Vec<usize> = (0..num_points).collect();
    order.sort_by(|&a, &b| densities[b].total_cmp(&densities[a]).then(a.cmp(&b)));

    // 3. delta_i and nearest higher-density neighbour
    let mut deltas = vec![0.0f32; num_points];
    let mut nearest_higher = vec![order[0]; num_points];
    for rank in 1..num_points {
        let idx = order[rank];
        let (best, d2) = order[..rank]
            .iter()
            .map(|&h| (h, dist2(row(idx), row(h))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("rank >= 1");
        deltas[idx] = d2.sqrt();
        nearest_higher[idx] = best;
    }

    // The global density maximum gets a delta above every other one
    let max_other_delta = order[1..].iter().map(|&i| deltas[i]).fold(0.0f32, f32::max);
    deltas[order[0]] = max_other_delta.max(1e-4) * 1.1;

    // 4. Select top `k` cluster centers by gamma_i = rho_i * delta_i
    let k_clusters = num_clusters.min(num_points);
    let mut gamma_order: Vec<usize> = (0..num_points).collect();
    gamma_order.sort_by(|&a, &b| (densities[b] * deltas[b]).total_cmp(&(densities[a] * deltas[a])).then(a.cmp(&b)));
    let cluster_centers: Vec<usize> = gamma_order[..k_clusters].to_vec();
    let mut labels = vec![usize::MAX; num_points];
    for (cid, &center_idx) in cluster_centers.iter().enumerate() {
        labels[center_idx] = cid;
    }

    // 5. Propagate cluster labels in descending density order
    for &idx in &order {
        if labels[idx] == usize::MAX {
            labels[idx] = labels[nearest_higher[idx]];
        }
    }

    DensityPeaksResult { labels, cluster_centers, densities, deltas }
}

/// Density peaks on at most `max_points` points: when `num_points` is larger, an evenly strided
/// subsample is clustered with [`cluster_density_peaks`] and every other point takes the label of
/// its nearest subsampled point. Time O(M²·D + N·M·D) for M = `max_points`; memory O(N).
///
/// `cluster_centers` index the full input; `densities` / `deltas` are NaN for points outside the
/// subsample.
pub fn cluster_density_peaks_capped(
    features: &[f32],
    num_points: usize,
    dim: usize,
    cutoff_distance: f32,
    num_clusters: usize,
    max_points: usize,
) -> DensityPeaksResult {
    let m = max_points.max(num_clusters).max(1);
    if num_points <= m {
        return cluster_density_peaks(features, num_points, dim, cutoff_distance, num_clusters);
    }
    assert_eq!(features.len(), num_points * dim);
    let picked: Vec<usize> = (0..m).map(|i| i * num_points / m).collect();
    let sub: Vec<f32> = picked.iter().flat_map(|&i| features[i * dim..(i + 1) * dim].iter().copied()).collect();
    let res = cluster_density_peaks(&sub, m, dim, cutoff_distance, num_clusters);

    let mut labels = vec![0usize; num_points];
    let mut densities = vec![f32::NAN; num_points];
    let mut deltas = vec![f32::NAN; num_points];
    for (s, &i) in picked.iter().enumerate() {
        densities[i] = res.densities[s];
        deltas[i] = res.deltas[s];
    }
    for (i, label) in labels.iter_mut().enumerate() {
        let x = &features[i * dim..(i + 1) * dim];
        let nearest = (0..m).min_by(|&a, &b| dist2(x, &sub[a * dim..(a + 1) * dim]).total_cmp(&dist2(x, &sub[b * dim..(b + 1) * dim])));
        *label = res.labels[nearest.expect("m >= 1")];
    }
    DensityPeaksResult {
        labels,
        cluster_centers: res.cluster_centers.iter().map(|&s| picked[s]).collect(),
        densities,
        deltas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_peaks_separates_three_clusters() {
        let centers = [[-10.0f32, -10.0], [0.0, 12.0], [14.0, -5.0]];
        let mut features = Vec::new();
        let mut expected_cluster = Vec::new();

        for (cid, c) in centers.iter().enumerate() {
            for k in 0..15 {
                let offset_x = ((k % 5) as f32 - 2.0) * 0.25;
                let offset_y = ((k / 5) as f32 - 1.0) * 0.25;
                features.push(c[0] + offset_x);
                features.push(c[1] + offset_y);
                expected_cluster.push(cid);
            }
        }

        let res = cluster_density_peaks(&features, 45, 2, 1.5, 3);
        assert_eq!(res.cluster_centers.len(), 3);

        // Verify all 15 points within each group share the same label and groups are distinct
        let l0 = res.labels[0];
        let l1 = res.labels[15];
        let l2 = res.labels[30];
        assert_ne!(l0, l1);
        assert_ne!(l1, l2);
        assert_ne!(l0, l2);

        for i in 0..15 {
            assert_eq!(res.labels[i], l0);
            assert_eq!(res.labels[15 + i], l1);
            assert_eq!(res.labels[30 + i], l2);
        }
    }

    #[test]
    fn capped_version_agrees_with_exact_on_separated_clusters() {
        let centers = [[-10.0f32, -10.0], [0.0, 12.0], [14.0, -5.0]];
        let mut features = Vec::new();
        for c in &centers {
            for k in 0..400 {
                features.push(c[0] + ((k % 20) as f32 - 10.0) * 0.05);
                features.push(c[1] + ((k / 20) as f32 - 10.0) * 0.05);
            }
        }
        let exact = cluster_density_peaks(&features, 1_200, 2, 1.5, 3);
        let capped = cluster_density_peaks_capped(&features, 1_200, 2, 1.5, 3, 150);
        // Same partition (labels may be permuted).
        for i in 0..1_200 {
            for j in [0, 400, 800] {
                assert_eq!(exact.labels[i] == exact.labels[j], capped.labels[i] == capped.labels[j]);
            }
        }
        assert!(capped.cluster_centers.iter().all(|&c| c < 1_200));
    }
}
