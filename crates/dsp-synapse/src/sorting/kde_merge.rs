//! KDE valley merge clustering: a heuristic in the spirit of MountainSort's IsoSplit (which tests
//! unimodality with isotonic regression; this does not).
//!
//! Starts from an over-clustering ($K_0$ k-means parcels) and repeatedly projects the closest pair
//! of clusters onto their centroid axis $\mathbf{v} = \boldsymbol{\mu}_B - \boldsymbol{\mu}_A$. A
//! Gaussian KDE of the projections is compared between the two centroids: the valley-to-peak ratio
//! gives `dip_score = 3 · (1 − ratio)`. Below `dip_threshold` the pair merges; otherwise the points
//! are re-cut at the lowest-density point. No significance test is made.

use serde::{Deserialize, Serialize};

/// Merge / re-cut passes before stopping.
const MAX_PASSES: usize = 30;
/// Lloyd iterations of the initial over-clustering.
const SEED_KMEANS_ITERATIONS: usize = 10;
/// Squared centroid distance below which two clusters are the same (merged without a test).
const SAME_CENTROID_DIST_SQ: f32 = 1e-10;
/// Fewest projected points, and smallest centroid distance, a valley test is made on.
const MIN_PROJECTED_POINTS: usize = 6;
const MIN_CENTROID_DISTANCE: f32 = 1e-6;
/// KDE bandwidth at least this fraction of the centroid distance.
const MIN_BANDWIDTH_FRACTION: f32 = 0.25;
/// Density below which a centroid has no peak to compare with.
const MIN_PEAK_DENSITY: f32 = 1e-8;
/// The valley is searched at `VALLEY_SEARCH` fractions of the centroid distance.
const VALLEY_SEARCH: [f32; 7] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
/// `dip_score = DIP_SCALE · (1 − valley / peak)` (valley / peak capped at `MAX_BRIDGE_RATIO`).
const DIP_SCALE: f32 = 3.0;
const MAX_BRIDGE_RATIO: f32 = 2.0;
/// Cut used when no valley test can be made: midway.
const MIDPOINT: f32 = 0.5;

/// Result of [`cluster_kde_merge`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KdeMergeResult {
    pub labels: Vec<i32>,
    pub num_clusters: usize,
    pub centroids: Vec<f32>,
}

/// Clusters `features` (`[num_spikes, num_features]`) by KDE valley merging (see the module docs).
pub fn cluster_kde_merge(
    features: &[f32],
    num_spikes: usize,
    num_features: usize,
    initial_k: usize,
    dip_threshold: f32,
    min_cluster_size: usize,
) -> KdeMergeResult {
    assert_eq!(features.len(), num_spikes * num_features);
    if num_spikes == 0 || num_features == 0 {
        return KdeMergeResult {
            labels: Vec::new(),
            num_clusters: 0,
            centroids: Vec::new(),
        };
    }

    let k0 = initial_k.clamp(1, (num_spikes / min_cluster_size.max(2)).max(1));
    let mut labels = initial_kmeans_partition(features, num_spikes, num_features, k0);

    for _pass in 0..MAX_PASSES {
        let (centroids, counts) = compute_centroids(features, &labels, num_spikes, num_features);
        let active_k = counts.len();
        if active_k <= 1 {
            break;
        }

        // Find closest pair of active clusters
        let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
        for a in 0..active_k {
            if counts[a] == 0 {
                continue;
            }
            let ca = &centroids[a * num_features..(a + 1) * num_features];
            for b in (a + 1)..active_k {
                if counts[b] == 0 {
                    continue;
                }
                let cb = &centroids[b * num_features..(b + 1) * num_features];
                let d2: f32 = ca.iter().zip(cb).map(|(x, y)| (x - y) * (x - y)).sum();
                pairs.push((a, b, d2));
            }
        }
        pairs.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal));

        // One merge or one boundary change per pass, then centroids are recomputed
        let mut changed = false;
        for (a, b, dist_sq) in pairs {
            if dist_sq < SAME_CENTROID_DIST_SQ {
                for l in &mut labels {
                    if *l == b as i32 {
                        *l = a as i32;
                    }
                }
                compact_labels(&mut labels);
                changed = true;
                break;
            }

            let ca = &centroids[a * num_features..(a + 1) * num_features];
            let cb = &centroids[b * num_features..(b + 1) * num_features];
            let inv_norm = 1.0 / dist_sq.sqrt();
            let dir: Vec<f32> = (0..num_features).map(|f| (cb[f] - ca[f]) * inv_norm).collect();

            // Project the points of a and b onto `dir`: dot(x − ca, dir) (ca at 0, cb at dist)
            let dist = dist_sq.sqrt();
            let proj_idx: Vec<(usize, f32)> = (0..num_spikes)
                .filter(|&i| labels[i] == a as i32 || labels[i] == b as i32)
                .map(|i| {
                    let xi = &features[i * num_features..(i + 1) * num_features];
                    (i, xi.iter().zip(ca).zip(&dir).map(|((x, c), v)| (x - c) * v).sum())
                })
                .collect();

            let (dip_score, cut_val) = evaluate_1d_dip(&proj_idx, dist);
            if dip_score < dip_threshold {
                // Unimodal along connecting axis -> merge cluster b into cluster a
                for (idx, _) in proj_idx {
                    labels[idx] = a as i32;
                }
                compact_labels(&mut labels);
                changed = true;
                break;
            } else {
                // Bimodal -> refine boundary at minimum-density cut point if it changes labels
                let mut count_left = 0usize;
                let mut count_right = 0usize;
                for &(_, p) in &proj_idx {
                    if p <= cut_val {
                        count_left += 1;
                    } else {
                        count_right += 1;
                    }
                }
                if count_left >= min_cluster_size && count_right >= min_cluster_size {
                    let mut moved = false;
                    for &(idx, p) in &proj_idx {
                        let side = if p <= cut_val { a as i32 } else { b as i32 };
                        moved |= labels[idx] != side;
                        labels[idx] = side;
                    }
                    if moved {
                        changed = true;
                        break;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    compact_labels(&mut labels);
    let (centroids, counts) = compute_centroids(features, &labels, num_spikes, num_features);
    KdeMergeResult {
        labels,
        num_clusters: counts.len(),
        centroids,
    }
}

/// Evaluates the 1D valley/dip depth between `0.0` (centroid A) and `dist` (centroid B)
/// using a smooth 1D Gaussian Kernel Density Estimate (KDE).
/// Returns `(dip_score, cut_coordinate)`.
fn evaluate_1d_dip(proj_idx: &[(usize, f32)], dist: f32) -> (f32, f32) {
    let n = proj_idx.len();
    if n < MIN_PROJECTED_POINTS || dist <= MIN_CENTROID_DISTANCE {
        return (0.0, dist * MIDPOINT);
    }

    // Estimate within-cluster 1D spread around 0 and dist
    let mut sum_sq = 0.0f32;
    for &(_, p) in proj_idx {
        let d_near = p.abs().min((p - dist).abs());
        sum_sq += d_near * d_near;
    }
    let sigma = (sum_sq / (n as f32)).sqrt().max(dist * MIN_BANDWIDTH_FRACTION);
    let inv_two_h2 = 1.0 / (2.0 * sigma * sigma);

    let kde = |x: f32| -> f32 {
        proj_idx
            .iter()
            .map(|&(_, p)| {
                let d = x - p;
                (-d * d * inv_two_h2).exp()
            })
            .sum()
    };

    let rho_a = kde(0.0);
    let rho_b = kde(dist);
    let ref_peak = rho_a.min(rho_b);
    if ref_peak <= MIN_PEAK_DENSITY {
        return (0.0, dist * MIDPOINT);
    }

    let mut min_valley = f32::INFINITY;
    let mut cut_coord = dist * MIDPOINT;
    for fraction in VALLEY_SEARCH {
        let x = dist * fraction;
        let r = kde(x);
        if r < min_valley {
            min_valley = r;
            cut_coord = x;
        }
    }

    let bridge_ratio = (min_valley / ref_peak).clamp(0.0, MAX_BRIDGE_RATIO);
    // When bridge_ratio is close to 1.0, there is no valley (unimodal -> dip_score ~ 0).
    // When bridge_ratio << 0.5, there is a deep valley (bimodal -> dip_score > 1.5).
    let dip_score = (1.0 - bridge_ratio).max(0.0) * DIP_SCALE;
    (dip_score, cut_coord)
}

fn initial_kmeans_partition(features: &[f32], n: usize, d: usize, k: usize) -> Vec<i32> {
    let mut centroids = vec![0.0f32; k * d];
    for c in 0..k {
        let src = (c * n) / k;
        centroids[c * d..(c + 1) * d].copy_from_slice(&features[src * d..(src + 1) * d]);
    }

    let mut labels = vec![0i32; n];
    for _ in 0..SEED_KMEANS_ITERATIONS {
        let mut sums = vec![0.0f32; k * d];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let xi = &features[i * d..(i + 1) * d];
            let mut best_c = 0;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let mc = &centroids[c * d..(c + 1) * d];
                let d2: f32 = xi.iter().zip(mc).map(|(a, b)| (a - b) * (a - b)).sum();
                if d2 < best_d {
                    best_d = d2;
                    best_c = c;
                }
            }
            labels[i] = best_c as i32;
            counts[best_c] += 1;
            for f in 0..d {
                sums[best_c * d + f] += xi[f];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / (counts[c] as f32);
                for f in 0..d {
                    centroids[c * d + f] = sums[c * d + f] * inv;
                }
            }
        }
    }
    compact_labels(&mut labels);
    labels
}

fn compact_labels(labels: &mut [i32]) {
    let mut unique: Vec<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
    unique.sort_unstable();
    unique.dedup();
    for l in labels.iter_mut() {
        if *l >= 0 {
            if let Ok(pos) = unique.binary_search(l) {
                *l = pos as i32;
            }
        }
    }
}

fn compute_centroids(
    features: &[f32],
    labels: &[i32],
    n: usize,
    d: usize,
) -> (Vec<f32>, Vec<usize>) {
    let max_l = labels.iter().copied().max().unwrap_or(-1);
    if max_l < 0 {
        return (Vec::new(), Vec::new());
    }
    let k = (max_l + 1) as usize;
    let mut centroids = vec![0.0f32; k * d];
    let mut counts = vec![0usize; k];

    for i in 0..n {
        let l = labels[i];
        if l >= 0 {
            let c = l as usize;
            counts[c] += 1;
            for f in 0..d {
                centroids[c * d + f] += features[i * d + f];
            }
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / (counts[c] as f32);
            for f in 0..d {
                centroids[c * d + f] *= inv;
            }
        }
    }
    (centroids, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kde_merge_merges_overclustered_parcels_into_three_true_clusters() {
        let centers = [[-12.0f32, 0.0], [0.0, 12.0], [12.0, 0.0]];
        let n_per = 60;
        let n = 3 * n_per;
        let d = 2;
        let mut features = Vec::with_capacity(n * d);

        let mut state = 0x2468_ACE0u64;
        let mut next_norm = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((state >> 32) as f32 / (u32::MAX as f32)).clamp(1e-6, 1.0 - 1e-6);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (state >> 32) as f32 / (u32::MAX as f32);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        };

        for center in &centers {
            for _ in 0..n_per {
                // Elongated non-spherical clusters
                let u = next_norm();
                let v = next_norm();
                features.push(center[0] + 1.2 * u);
                features.push(center[1] + 0.5 * v);
            }
        }

        // Start with K0 = 9 over-clustered parcels; unimodal sub-parcels merge down to 3
        let res = cluster_kde_merge(&features, n, d, 9, 1.5, 5);
        assert_eq!(res.num_clusters, 3, "expected 3 clusters, got {}", res.num_clusters);
    }
}
