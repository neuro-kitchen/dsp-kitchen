//! Feature-Space Cluster Separation & Isolation Metrics (`isolation.rs`).
//!
//! Provides the standard PCA/latent feature space cluster isolation metrics:
//! - **$d'$ (Linear Discriminant Sensitivity)**: Hill et al. (2011) Fisher LDA separation.
//! - **Silhouette Score**: Rousseeuw (1987) intra- vs. inter-cluster distance ratio in `[-1.0, 1.0]`.
//! - **Isolation Distance**: Schmitzer-Torbert et al. (2005) $N_C$-th closest outside spike Mahalanobis distance.

/// Computes Fisher's Linear Discriminant sensitivity $d'$ between `cluster_a` (`[N_a, D]`)
/// and `cluster_b` (`[N_b, D]`).
pub fn compute_d_prime(cluster_a: &[f32], n_a: usize, cluster_b: &[f32], n_b: usize, dim: usize) -> f32 {
    if n_a < 2 || n_b < 2 || dim == 0 {
        return 0.0;
    }
    assert_eq!(cluster_a.len(), n_a * dim);
    assert_eq!(cluster_b.len(), n_b * dim);

    let mut mean_a = vec![0.0f64; dim];
    let mut mean_b = vec![0.0f64; dim];

    for i in 0..n_a {
        for d in 0..dim {
            mean_a[d] += cluster_a[i * dim + d] as f64;
        }
    }
    for d in 0..dim {
        mean_a[d] /= n_a as f64;
    }

    for i in 0..n_b {
        for d in 0..dim {
            mean_b[d] += cluster_b[i * dim + d] as f64;
        }
    }
    for d in 0..dim {
        mean_b[d] /= n_b as f64;
    }

    // Diagonal pooled variance & Fisher axis w = (mu_a - mu_b) / sigma_pooled^2
    let mut var_a = vec![0.0f64; dim];
    let mut var_b = vec![0.0f64; dim];
    for i in 0..n_a {
        for d in 0..dim {
            let diff = (cluster_a[i * dim + d] as f64) - mean_a[d];
            var_a[d] += diff * diff;
        }
    }
    for i in 0..n_b {
        for d in 0..dim {
            let diff = (cluster_b[i * dim + d] as f64) - mean_b[d];
            var_b[d] += diff * diff;
        }
    }

    let mut axis = vec![0.0f64; dim];
    for d in 0..dim {
        let pooled_var = 0.5 * (var_a[d] / (n_a as f64) + var_b[d] / (n_b as f64)).max(1e-9);
        axis[d] = (mean_a[d] - mean_b[d]) / pooled_var;
    }

    let proj_point = |row: &[f32]| -> f64 {
        let mut s = 0.0f64;
        for d in 0..dim {
            s += (row[d] as f64) * axis[d];
        }
        s
    };

    let proj_a: Vec<f64> = (0..n_a)
        .map(|i| proj_point(&cluster_a[i * dim..(i + 1) * dim]))
        .collect();
    let proj_b: Vec<f64> = (0..n_b)
        .map(|i| proj_point(&cluster_b[i * dim..(i + 1) * dim]))
        .collect();

    let mu_pa = proj_a.iter().sum::<f64>() / (n_a as f64);
    let mu_pb = proj_b.iter().sum::<f64>() / (n_b as f64);

    let v_pa = proj_a
        .iter()
        .map(|&v| (v - mu_pa) * (v - mu_pa))
        .sum::<f64>()
        / (n_a as f64);
    let v_pb = proj_b
        .iter()
        .map(|&v| (v - mu_pb) * (v - mu_pb))
        .sum::<f64>()
        / (n_b as f64);

    let denom = (0.5 * (v_pa + v_pb)).sqrt().max(1e-9);
    ((mu_pa - mu_pb).abs() / denom) as f32
}

/// Computes the Silhouette Score in `[-1.0, 1.0]` for `target_unit` against all other labeled spikes.
pub fn compute_silhouette_score(
    features: &[f32],
    labels: &[usize],
    dim: usize,
    target_unit: usize,
) -> f32 {
    let n = labels.len();
    if n < 2 || dim == 0 {
        return 0.0;
    }
    assert_eq!(features.len(), n * dim);

    let target_indices: Vec<usize> = (0..n).filter(|&i| labels[i] == target_unit).collect();
    if target_indices.len() < 2 {
        return 0.0;
    }

    let mut other_units: Vec<usize> = labels
        .iter()
        .copied()
        .filter(|&u| u != target_unit)
        .collect();
    other_units.sort_unstable();
    other_units.dedup();
    if other_units.is_empty() {
        return 1.0;
    }

    let dist = |i: usize, j: usize| -> f32 {
        let ri = &features[i * dim..(i + 1) * dim];
        let rj = &features[j * dim..(j + 1) * dim];
        let mut d2 = 0.0f32;
        for d in 0..dim {
            let diff = ri[d] - rj[d];
            d2 += diff * diff;
        }
        d2.sqrt()
    };

    let mut total_s = 0.0f32;
    for &idx in &target_indices {
        let mut sum_intra = 0.0f32;
        for &other_in in &target_indices {
            if other_in != idx {
                sum_intra += dist(idx, other_in);
            }
        }
        let a_i = sum_intra / ((target_indices.len() - 1) as f32);

        let mut b_i = f32::INFINITY;
        for &u_other in &other_units {
            let mut sum_inter = 0.0f32;
            let mut count_inter = 0usize;
            for j in 0..n {
                if labels[j] == u_other {
                    sum_inter += dist(idx, j);
                    count_inter += 1;
                }
            }
            if count_inter > 0 {
                let mean_inter = sum_inter / (count_inter as f32);
                if mean_inter < b_i {
                    b_i = mean_inter;
                }
            }
        }

        let denom = a_i.max(b_i).max(1e-8);
        total_s += (b_i - a_i) / denom;
    }

    total_s / (target_indices.len() as f32)
}

/// Computes the Schmitzer-Torbert et al. (2005) Mahalanobis Isolation Distance
/// for `target_unit` in `[N, D]` feature space.
pub fn compute_isolation_distance(
    features: &[f32],
    labels: &[usize],
    dim: usize,
    target_unit: usize,
) -> f32 {
    let n = labels.len();
    if n < 4 || dim == 0 {
        return 0.0;
    }
    assert_eq!(features.len(), n * dim);

    let target_indices: Vec<usize> = (0..n).filter(|&i| labels[i] == target_unit).collect();
    let other_indices: Vec<usize> = (0..n).filter(|&i| labels[i] != target_unit).collect();
    let n_c = target_indices.len();

    if n_c < 2 || other_indices.len() < n_c {
        return 0.0;
    }

    let mut mean = vec![0.0f64; dim];
    for &i in &target_indices {
        for d in 0..dim {
            mean[d] += features[i * dim + d] as f64;
        }
    }
    for d in 0..dim {
        mean[d] /= n_c as f64;
    }

    let mut var = vec![0.0f64; dim];
    for &i in &target_indices {
        for d in 0..dim {
            let diff = (features[i * dim + d] as f64) - mean[d];
            var[d] += diff * diff;
        }
    }
    for d in 0..dim {
        var[d] = (var[d] / (n_c as f64)).max(1e-6);
    }

    let mut outside_d2: Vec<f32> = other_indices
        .iter()
        .map(|&j| {
            let mut m2 = 0.0f64;
            for d in 0..dim {
                let diff = (features[j * dim + d] as f64) - mean[d];
                m2 += (diff * diff) / var[d];
            }
            m2 as f32
        })
        .collect();

    outside_d2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    outside_d2[n_c - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_metrics_d_prime_silhouette_and_mahalanobis() {
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for i in 0..30 {
            let jitter = ((i % 5) as f32 - 2.0) * 0.2;
            features.push(-10.0 + jitter);
            features.push(-10.0 - jitter);
            labels.push(0);
        }
        for i in 0..30 {
            let jitter = ((i % 5) as f32 - 2.0) * 0.2;
            features.push(10.0 + jitter);
            features.push(10.0 - jitter);
            labels.push(1);
        }

        let dp = compute_d_prime(&features[0..60], 30, &features[60..120], 30, 2);
        assert!(dp > 20.0, "d_prime = {}", dp);

        let sil = compute_silhouette_score(&features, &labels, 2, 0);
        assert!(sil > 0.90, "silhouette = {}", sil);

        let iso_dist = compute_isolation_distance(&features, &labels, 2, 0);
        assert!(iso_dist > 100.0, "iso_dist = {}", iso_dist);
    }
}
