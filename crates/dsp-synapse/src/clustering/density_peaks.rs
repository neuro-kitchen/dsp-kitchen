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

/// Runs Rodriguez-Laio Density Peaks clustering on a flat `[num_points, dim]` feature matrix.
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

    let dc_sq = (cutoff_distance.max(1e-4)) * (cutoff_distance.max(1e-4));

    // 1. Pairwise Euclidean distances and Gaussian local densities rho_i
    let mut dist_mat = vec![0.0f32; num_points * num_points];
    let mut densities = vec![0.0f32; num_points];
    let mut max_dist = 0.0f32;

    for i in 0..num_points {
        let row_i = &features[i * dim..(i + 1) * dim];
        for j in (i + 1)..num_points {
            let row_j = &features[j * dim..(j + 1) * dim];
            let mut d2 = 0.0f32;
            for k in 0..dim {
                let diff = row_i[k] - row_j[k];
                d2 += diff * diff;
            }
            let d = d2.sqrt();
            if d > max_dist {
                max_dist = d;
            }
            dist_mat[i * num_points + j] = d;
            dist_mat[j * num_points + i] = d;

            let kernel = (-d2 / dc_sq).exp();
            densities[i] += kernel;
            densities[j] += kernel;
        }
    }

    // 2. Sort point indices in descending order of density rho
    let mut order: Vec<usize> = (0..num_points).collect();
    order.sort_by(|&a, &b| {
        densities[b]
            .partial_cmp(&densities[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 3. Compute delta_i and nearest higher-density neighbor index
    let mut deltas = vec![0.0f32; num_points];
    let mut nearest_higher = vec![0usize; num_points];

    deltas[order[0]] = max_dist.max(1.0);
    nearest_higher[order[0]] = order[0];

    for rank in 1..num_points {
        let idx = order[rank];
        let mut min_d = f32::INFINITY;
        let mut best_parent = order[0];

        for higher_rank in 0..rank {
            let h_idx = order[higher_rank];
            let d = dist_mat[idx * num_points + h_idx];
            if d < min_d {
                min_d = d;
                best_parent = h_idx;
            }
        }
        deltas[idx] = min_d;
        nearest_higher[idx] = best_parent;
    }

    // Set global maximum density point's delta to maximum among all other deltas
    if num_points > 1 {
        let max_other_delta = order[1..]
            .iter()
            .map(|&i| deltas[i])
            .fold(0.0f32, f32::max);
        deltas[order[0]] = max_other_delta.max(1e-4) * 1.1;
    }

    // 4. Select top `k` cluster centers by gamma_i = rho_i * delta_i
    let k_clusters = num_clusters.min(num_points);
    let mut gamma_order: Vec<usize> = (0..num_points).collect();
    gamma_order.sort_by(|&a, &b| {
        let ga = densities[a] * deltas[a];
        let gb = densities[b] * deltas[b];
        gb.partial_cmp(&ga).unwrap_or(std::cmp::Ordering::Equal)
    });

    let cluster_centers: Vec<usize> = gamma_order[..k_clusters].to_vec();
    let mut labels = vec![usize::MAX; num_points];
    for (cid, &center_idx) in cluster_centers.iter().enumerate() {
        labels[center_idx] = cid;
    }

    // 5. Propagate cluster labels in descending density order
    for &idx in &order {
        if labels[idx] == usize::MAX {
            let parent = nearest_higher[idx];
            labels[idx] = labels[parent];
        }
    }

    DensityPeaksResult {
        labels,
        cluster_centers,
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
}
