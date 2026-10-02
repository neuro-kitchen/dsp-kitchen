//! Gaussian Mixture Model (GMM) Spike Clustering with EM, Masked EM, and BIC Selection (`gmm.rs`).
//!
//! Models PCA / PPCA / `wPCA` spike feature embeddings $\mathbf{x}_i \in \mathbb{R}^D$ as a mixture of
//! $K$ multivariate Gaussians:
//! $$p(\mathbf{x}_i) = \sum_{k=1}^K \pi_k \,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
//! Supports:
//! - Diagonal, Full, and **Masked EM** (KlustaKwik / Rossant et al. 2016 noise-prior masking for high-density probes)
//! - Deterministic `k-means++` / farthest-first seeding
//! - Automatic cluster count selection $K^* = \arg\min_{K \in [K_{\min}, K_{\max}]} \text{BIC}(K)$
//! - Soft posterior assignment probabilities $p(z_i = k \mid \mathbf{x}_i)$ and Mahalanobis refractory/outlier gating.

use dsp_base::linalg::SymmetricEig;
use serde::{Deserialize, Serialize};

/// Covariance structure for Gaussian Mixture Model EM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GmmCovarianceKind {
    /// Diagonal covariance per cluster ($\boldsymbol{\Sigma}_k = \text{diag}(\sigma_{k,1}^2, \dots, \sigma_{k,D}^2)$).
    Diagonal,
    /// Full symmetric positive-definite covariance matrix per cluster ($\boldsymbol{\Sigma}_k \in \mathbb{R}^{D \times D}$).
    #[default]
    Full,
    /// Masked EM (KlustaKwik style): unmasked dimensions follow cluster statistics while masked
    /// background dimensions shrink toward a shared unit-variance noise prior $\mathcal{N}(0, 1)$.
    Masked,
}

/// Output of Gaussian Mixture Model (GMM) spike clustering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GmmResult {
    /// Hard cluster label for each spike (`0..num_clusters`, or `-1` if gated as an outlier).
    pub labels: Vec<i32>,
    /// Number of selected Gaussian clusters $K$.
    pub num_clusters: usize,
    /// Mixture weights $\pi_k$ of length `num_clusters`.
    pub weights: Vec<f32>,
    /// Cluster means $\boldsymbol{\mu}_k$ of shape `[num_clusters, num_features]`.
    pub means: Vec<f32>,
    /// Row-major cluster covariances of shape `[num_clusters, num_features, num_features]`.
    pub covariances: Vec<f32>,
    /// Soft posterior responsibilities $r_{ik} = p(z_i = k \mid \mathbf{x}_i)$ of shape `[num_spikes, num_clusters]`.
    pub responsibilities: Vec<f32>,
    /// Squared Mahalanobis distance $d_M^2(\mathbf{x}_i, \boldsymbol{\mu}_{z_i})$ of each spike to its assigned cluster.
    pub mahalanobis_sq: Vec<f32>,
    /// Final log-likelihood $\sum_{i=1}^N \ln p(\mathbf{x}_i)$.
    pub log_likelihood: f64,
    /// Bayesian Information Criterion $\text{BIC} = -2 \ln \mathcal{L} + p \ln N$ (lower is better).
    pub bic: f64,
}

/// Configurable Gaussian Mixture Model (GMM) spike clusterer.
#[derive(Debug, Clone)]
pub struct GmmClusterer {
    pub k_min: usize,
    pub k_max: usize,
    pub covariance_kind: GmmCovarianceKind,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub regularization: f32,
    /// Optional squared Mahalanobis outlier threshold (e.g. `Some(25.0)` labels spikes with $d_M > 5\sigma$ as `-1`).
    pub outlier_mahalanobis_sq: Option<f32>,
}

impl Default for GmmClusterer {
    fn default() -> Self {
        Self {
            k_min: 1,
            k_max: 8,
            covariance_kind: GmmCovarianceKind::Full,
            max_iterations: 100,
            tolerance: 1e-5,
            regularization: 1e-4,
            outlier_mahalanobis_sq: None,
        }
    }
}

impl GmmClusterer {
    pub fn new(k_min: usize, k_max: usize, covariance_kind: GmmCovarianceKind) -> Self {
        Self {
            k_min: k_min.max(1),
            k_max: k_max.max(k_min.max(1)),
            covariance_kind,
            ..Self::default()
        }
    }

    /// Fits GMM for a fixed cluster count `k`.
    pub fn fit_k(
        &self,
        features: &[f32],
        num_spikes: usize,
        num_features: usize,
        k: usize,
        feature_mask: Option<&[f32]>,
    ) -> GmmResult {
        fit_gmm_single_k(
            features,
            num_spikes,
            num_features,
            k.max(1),
            self.covariance_kind,
            self.max_iterations,
            self.tolerance,
            self.regularization,
            self.outlier_mahalanobis_sq,
            feature_mask,
        )
    }

    /// Sweeps $K \in [k_{\min}, k_{\max}]$ and returns the model minimizing the Bayesian Information Criterion (BIC).
    pub fn fit(
        &self,
        features: &[f32],
        num_spikes: usize,
        num_features: usize,
        feature_mask: Option<&[f32]>,
    ) -> GmmResult {
        if num_spikes == 0 || num_features == 0 {
            return GmmResult {
                labels: Vec::new(),
                num_clusters: 0,
                weights: Vec::new(),
                means: Vec::new(),
                covariances: Vec::new(),
                responsibilities: Vec::new(),
                mahalanobis_sq: Vec::new(),
                log_likelihood: 0.0,
                bic: 0.0,
            };
        }
        let k_lo = self.k_min.clamp(1, num_spikes);
        let k_hi = self.k_max.clamp(k_lo, num_spikes);

        let mut best: Option<GmmResult> = None;
        for k in k_lo..=k_hi {
            let candidate = self.fit_k(features, num_spikes, num_features, k, feature_mask);
            if best.as_ref().map_or(true, |b| candidate.bic < b.bic) {
                best = Some(candidate);
            }
        }
        best.unwrap()
    }
}

/// Convenience function to fit a GMM with automatic BIC selection over `k_min..=k_max`.
pub fn cluster_gmm_bic(
    features: &[f32],
    num_spikes: usize,
    num_features: usize,
    k_min: usize,
    k_max: usize,
) -> GmmResult {
    GmmClusterer::new(k_min, k_max, GmmCovarianceKind::Full).fit(features, num_spikes, num_features, None)
}

#[allow(clippy::too_many_arguments)]
fn fit_gmm_single_k(
    features: &[f32],
    n: usize,
    d: usize,
    k: usize,
    cov_kind: GmmCovarianceKind,
    max_iters: usize,
    tol: f64,
    reg: f32,
    outlier_mahal_sq: Option<f32>,
    feature_mask: Option<&[f32]>,
) -> GmmResult {
    assert_eq!(features.len(), n * d);
    let k = k.clamp(1, n);
    let reg = reg.max(1e-6);

    // 1. Deterministic farthest-first / k-means++ initialization + 5 Lloyd steps
    let mut means = initialize_centroids_diverse(features, n, d, k);
    let mut weights = vec![1.0f32 / (k as f32); k];
    let mut covariances = vec![0.0f32; k * d * d];
    for c in 0..k {
        for f in 0..d {
            covariances[c * d * d + f * d + f] = 1.0;
        }
    }

    // Initialize cluster covariances from nearest-centroid partition
    let mut init_counts = vec![0usize; k];
    for i in 0..n {
        let xi = &features[i * d..(i + 1) * d];
        let mut best_c = 0;
        let mut best_dist = f32::INFINITY;
        for c in 0..k {
            let mc = &means[c * d..(c + 1) * d];
            let dist: f32 = xi.iter().zip(mc).map(|(a, b)| (a - b) * (a - b)).sum();
            if dist < best_dist {
                best_dist = dist;
                best_c = c;
            }
        }
        init_counts[best_c] += 1;
        let mc = &means[best_c * d..(best_c + 1) * d];
        let cov_c = &mut covariances[best_c * d * d..(best_c + 1) * d * d];
        for r in 0..d {
            let dr = xi[r] - mc[r];
            for col in 0..d {
                let dc = xi[col] - mc[col];
                if cov_kind == GmmCovarianceKind::Diagonal && r != col {
                    continue;
                }
                cov_c[r * d + col] += dr * dc;
            }
        }
    }
    for c in 0..k {
        let cnt = (init_counts[c] as f32).max(1.0);
        weights[c] = (cnt / (n as f32)).max(1e-4);
        let cov_c = &mut covariances[c * d * d..(c + 1) * d * d];
        for elem in cov_c.iter_mut() {
            *elem /= cnt;
        }
        for f in 0..d {
            cov_c[f * d + f] = cov_c[f * d + f].max(reg) + reg;
        }
    }

    let mut responsibilities = vec![0.0f32; n * k];
    let mut mahalanobis_sq = vec![0.0f32; n];
    let mut prev_ll = f64::NEG_INFINITY;
    let mut log_likelihood = f64::NEG_INFINITY;
    let ln_2pi_d = (d as f64) * (2.0 * std::f64::consts::PI).ln();

    // Precompute precision matrices (Sigma_k^-1) and log determinants
    let mut precisions = vec![0.0f32; k * d * d];
    let mut log_dets = vec![0.0f64; k];
    let mut log_probs = vec![0.0f64; k];
    let mut new_mean = vec![0.0f32; d];
    let mut mask_sum = vec![0.0f32; d];

    for _iter in 0..max_iters.max(1) {
        if cov_kind == GmmCovarianceKind::Diagonal {
            for c in 0..k {
                let cov_c = &covariances[c * d * d..(c + 1) * d * d];
                let prec_c = &mut precisions[c * d * d..(c + 1) * d * d];
                prec_c.fill(0.0);
                let mut ldet = 0.0f64;
                for f in 0..d {
                    let v = (cov_c[f * d + f]).max(reg);
                    ldet += (v as f64).ln();
                    prec_c[f * d + f] = 1.0 / v;
                }
                log_dets[c] = ldet;
            }
        } else {
            for c in 0..k {
                let cov_c = &covariances[c * d * d..(c + 1) * d * d];
                let (inv_c, ldet) = invert_spd_and_logdet(cov_c, d, reg);
                precisions[c * d * d..(c + 1) * d * d].copy_from_slice(&inv_c);
                log_dets[c] = ldet;
            }
        }

        // E-step: evaluate log responsibilities
        let mut total_ll = 0.0f64;
        for i in 0..n {
            let xi = &features[i * d..(i + 1) * d];
            let mut max_lp = f64::NEG_INFINITY;

            for c in 0..k {
                let mc = &means[c * d..(c + 1) * d];
                let prec_c = &precisions[c * d * d..(c + 1) * d * d];
                let d_m2 = if cov_kind == GmmCovarianceKind::Diagonal {
                    let mut sum = 0.0f32;
                    for f in 0..d {
                        let diff = xi[f] - mc[f];
                        sum += diff * diff * prec_c[f * d + f];
                    }
                    sum
                } else {
                    quad_form_mahalanobis(xi, mc, prec_c, d)
                };
                let lp = (weights[c].max(1e-12) as f64).ln() - 0.5 * (ln_2pi_d + log_dets[c] + d_m2 as f64);
                log_probs[c] = lp;
                if lp > max_lp {
                    max_lp = lp;
                }
            }

            let mut sum_exp = 0.0f64;
            for c in 0..k {
                let e = (log_probs[c] - max_lp).exp();
                responsibilities[i * k + c] = e as f32;
                sum_exp += e;
            }
            let inv_sum = (1.0 / sum_exp.max(1e-30)) as f32;
            for c in 0..k {
                responsibilities[i * k + c] *= inv_sum;
            }
            total_ll += max_lp + sum_exp.ln();
        }

        log_likelihood = total_ll;
        if (log_likelihood - prev_ll).abs() < tol {
            break;
        }
        prev_ll = log_likelihood;

        // M-step: update weights, means, and covariances
        for c in 0..k {
            let mut nk = 0.0f64;
            for i in 0..n {
                nk += responsibilities[i * k + c] as f64;
            }
            let nk_safe = nk.max(1e-8) as f32;
            weights[c] = ((nk / (n as f64)) as f32).clamp(1e-5, 1.0);

            new_mean.fill(0.0);
            mask_sum.fill(0.0);
            for i in 0..n {
                let r_ic = responsibilities[i * k + c];
                if r_ic < 1e-9 {
                    continue;
                }
                let xi = &features[i * d..(i + 1) * d];
                for f in 0..d {
                    let m_if = if cov_kind == GmmCovarianceKind::Masked {
                        feature_mask
                            .map(|m| m[i * d + f].clamp(0.0, 1.0))
                            .unwrap_or(1.0)
                    } else {
                        1.0
                    };
                    new_mean[f] += r_ic * m_if * xi[f];
                    mask_sum[f] += r_ic * m_if;
                }
            }
            for f in 0..d {
                means[c * d + f] = new_mean[f] / nk_safe;
            }

            let mc = &means[c * d..(c + 1) * d];
            let cov_c = &mut covariances[c * d * d..(c + 1) * d * d];
            cov_c.fill(0.0);

            if cov_kind == GmmCovarianceKind::Diagonal {
                for i in 0..n {
                    let r_ic = responsibilities[i * k + c];
                    if r_ic < 1e-9 {
                        continue;
                    }
                    let xi = &features[i * d..(i + 1) * d];
                    for r in 0..d {
                        let dr = xi[r] - mc[r];
                        cov_c[r * d + r] += r_ic * dr * dr;
                    }
                }
            } else {
                for i in 0..n {
                    let r_ic = responsibilities[i * k + c];
                    if r_ic < 1e-9 {
                        continue;
                    }
                    let xi = &features[i * d..(i + 1) * d];
                    for r in 0..d {
                        let dr = xi[r] - mc[r];
                        for col in r..d {
                            let dc = xi[col] - mc[col];
                            let v = r_ic * dr * dc;
                            cov_c[r * d + col] += v;
                            if r != col {
                                cov_c[col * d + r] += v;
                            }
                        }
                    }
                }
            }

            for elem in cov_c.iter_mut() {
                *elem /= nk_safe;
            }
            for f in 0..d {
                if cov_kind == GmmCovarianceKind::Masked {
                    let obs_frac = (mask_sum[f] / nk_safe).clamp(0.0, 1.0);
                    cov_c[f * d + f] = obs_frac * cov_c[f * d + f] + (1.0 - obs_frac) * 1.0 + reg;
                } else {
                    cov_c[f * d + f] += reg;
                }
            }
        }
    }

    // Final hard labels and Mahalanobis distances
    if cov_kind == GmmCovarianceKind::Diagonal {
        for c in 0..k {
            let cov_c = &covariances[c * d * d..(c + 1) * d * d];
            let prec_c = &mut precisions[c * d * d..(c + 1) * d * d];
            prec_c.fill(0.0);
            let mut ldet = 0.0f64;
            for f in 0..d {
                let v = cov_c[f * d + f].max(reg);
                ldet += (v as f64).ln();
                prec_c[f * d + f] = 1.0 / v;
            }
            log_dets[c] = ldet;
        }
    } else {
        for c in 0..k {
            let cov_c = &covariances[c * d * d..(c + 1) * d * d];
            let (inv_c, ldet) = invert_spd_and_logdet(cov_c, d, reg);
            precisions[c * d * d..(c + 1) * d * d].copy_from_slice(&inv_c);
            log_dets[c] = ldet;
        }
    }

    let mut labels = vec![0i32; n];
    for i in 0..n {
        let mut best_c = 0usize;
        let mut best_r = -1.0f32;
        for c in 0..k {
            let r = responsibilities[i * k + c];
            if r > best_r {
                best_r = r;
                best_c = c;
            }
        }
        let xi = &features[i * d..(i + 1) * d];
        let mc = &means[best_c * d..(best_c + 1) * d];
        let prec_c = &precisions[best_c * d * d..(best_c + 1) * d * d];
        let d_m2 = if cov_kind == GmmCovarianceKind::Diagonal {
            let mut sum = 0.0f32;
            for f in 0..d {
                let diff = xi[f] - mc[f];
                sum += diff * diff * prec_c[f * d + f];
            }
            sum
        } else {
            quad_form_mahalanobis(xi, mc, prec_c, d)
        };
        mahalanobis_sq[i] = d_m2;
        labels[i] = if let Some(max_d2) = outlier_mahal_sq {
            if d_m2 > max_d2 { -1 } else { best_c as i32 }
        } else {
            best_c as i32
        };
    }

    // Number of free parameters p for BIC = -2 ln L + p ln N
    let cov_params_per_cluster = match cov_kind {
        GmmCovarianceKind::Diagonal => d,
        GmmCovarianceKind::Full | GmmCovarianceKind::Masked => (d * (d + 1)) / 2,
    };
    let num_params = (k - 1) + k * d + k * cov_params_per_cluster;
    let bic = -2.0 * log_likelihood + (num_params as f64) * (n as f64).max(1.0).ln();

    GmmResult {
        labels,
        num_clusters: k,
        weights,
        means,
        covariances,
        responsibilities,
        mahalanobis_sq,
        log_likelihood,
        bic,
    }
}

fn initialize_centroids_diverse(features: &[f32], n: usize, d: usize, k: usize) -> Vec<f32> {
    let mut centroids = vec![0.0f32; k * d];
    // Pick first centroid closest to overall mean, then farthest-first + local density refinement
    let mut global_mean = vec![0.0f32; d];
    for i in 0..n {
        for f in 0..d {
            global_mean[f] += features[i * d + f] / (n as f32);
        }
    }
    let mut first_idx = 0usize;
    let mut max_d0 = -1.0f32;
    for i in 0..n {
        let xi = &features[i * d..(i + 1) * d];
        let dist: f32 = xi.iter().zip(&global_mean).map(|(a, b)| (a - b) * (a - b)).sum();
        if dist > max_d0 {
            max_d0 = dist;
            first_idx = i;
        }
    }
    centroids[..d].copy_from_slice(&features[first_idx * d..(first_idx + 1) * d]);

    let mut min_dists = vec![f32::INFINITY; n];
    for c in 1..k {
        let prev = &centroids[(c - 1) * d..c * d];
        let mut best_i = 0usize;
        let mut best_dist = -1.0f32;
        for i in 0..n {
            let xi = &features[i * d..(i + 1) * d];
            let d2: f32 = xi.iter().zip(prev).map(|(a, b)| (a - b) * (a - b)).sum();
            if d2 < min_dists[i] {
                min_dists[i] = d2;
            }
            if min_dists[i] > best_dist {
                best_dist = min_dists[i];
                best_i = i;
            }
        }
        centroids[c * d..(c + 1) * d].copy_from_slice(&features[best_i * d..(best_i + 1) * d]);
    }

    // Run 8 Lloyd k-means iterations to settle centroids in high-density cluster cores
    for _ in 0..8 {
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

    centroids
}

fn invert_spd_and_logdet(cov: &[f32], d: usize, reg: f32) -> (Vec<f32>, f64) {
    let eig = SymmetricEig::decompose(cov, d, 80);
    let mut inv = vec![0.0f32; d * d];
    let mut log_det = 0.0f64;

    let inv_evals: Vec<f32> = eig
        .eigenvalues
        .iter()
        .map(|&lam| {
            let safe_lam = lam.max(reg);
            log_det += (safe_lam as f64).ln();
            1.0 / safe_lam
        })
        .collect();

    for i in 0..d {
        for j in i..d {
            let mut sum = 0.0f32;
            for m in 0..d {
                sum += eig.eigenvectors[i * d + m] * inv_evals[m] * eig.eigenvectors[j * d + m];
            }
            inv[i * d + j] = sum;
            inv[j * d + i] = sum;
        }
    }
    (inv, log_det)
}

#[inline]
fn quad_form_mahalanobis(x: &[f32], mean: &[f32], precision: &[f32], d: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..d {
        let di = x[i] - mean[i];
        let row = &precision[i * d..(i + 1) * d];
        let mut inner = 0.0f32;
        for j in 0..d {
            inner += row[j] * (x[j] - mean[j]);
        }
        sum += di * inner;
    }
    sum.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gmm_bic_selects_three_clusters_and_recovers_labels() {
        let centers = [[-8.0f32, -6.0], [0.0, 7.0], [9.0, -4.0]];
        let n_per = 60;
        let n = 3 * n_per;
        let d = 2;
        let mut features = Vec::with_capacity(n * d);

        let mut state = 0x1357_9BDFu64;
        let mut next_norm = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((state >> 32) as f32 / (u32::MAX as f32)).clamp(1e-6, 1.0 - 1e-6);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (state >> 32) as f32 / (u32::MAX as f32);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        };

        for c in 0..3 {
            for _ in 0..n_per {
                features.push(centers[c][0] + 0.6 * next_norm());
                features.push(centers[c][1] + 0.6 * next_norm());
            }
        }

        let res = cluster_gmm_bic(&features, n, d, 1, 6);
        assert_eq!(res.num_clusters, 3, "BIC selected K={}", res.num_clusters);

        let l0 = res.labels[0];
        let l1 = res.labels[n_per];
        let l2 = res.labels[2 * n_per];
        assert_ne!(l0, l1);
        assert_ne!(l1, l2);
        assert_ne!(l0, l2);

        let mut correct = 0usize;
        for i in 0..n_per {
            if res.labels[i] == l0 {
                correct += 1;
            }
            if res.labels[n_per + i] == l1 {
                correct += 1;
            }
            if res.labels[2 * n_per + i] == l2 {
                correct += 1;
            }
        }
        assert!(correct as f32 / (n as f32) > 0.98);
    }
}
