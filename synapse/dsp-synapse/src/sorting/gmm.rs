//! Gaussian Mixture Model (GMM) Spike Clustering with EM, Masked EM, and BIC Selection (`gmm.rs`).
//!
//! Models PCA / PPCA / `wPCA` spike feature embeddings $\mathbf{x}_i \in \mathbb{R}^D$ as a mixture of
//! $K$ multivariate Gaussians:
//! $$p(\mathbf{x}_i) = \sum_{k=1}^K \pi_k \,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
//! Supports:
//! - Diagonal, Full, and **Masked EM** (KlustaKwik / Rossant et al. 2016 noise-prior masking for high-density probes)
//! - Deterministic farthest-first seeding (not k-means++: no random draws), refined by Lloyd steps
//! - Automatic cluster count selection $K^* = \arg\min_{K \in [K_{\min}, K_{\max}]} \text{BIC}(K)$
//! - Soft posterior assignment probabilities $p(z_i = k \mid \mathbf{x}_i)$ and Mahalanobis refractory/outlier gating.

use cubecl::prelude::*;
use dsp_base::core::{buffer, reduce};
use dsp_base::linalg::spd_inverse_logdet;
use dsp_core::compute::LaunchGeometry;

use super::kernels::{gmm_e_step_kernel, gmm_mean_sums_kernel, gmm_scatter_kernel};

/// Default EM tolerance on the mean per-spike log-likelihood (sklearn `GaussianMixture.tol`).
pub const DEFAULT_TOLERANCE: f64 = 1e-3;

/// Smallest ridge `reg` the fit uses.
const MIN_REGULARIZATION: f32 = 1e-6;
/// Smallest mixture weight of the initial (nearest-seed) partition.
const MIN_INIT_WEIGHT: f32 = 1e-4;
/// Lloyd iterations refining the farthest-first seeds.
const SEED_LLOYD_ITERATIONS: usize = 8;

/// Smallest mixture weight kept after an M-step (a component never fully vanishes).
const MIN_WEIGHT: f32 = 1e-5;
/// Weight floor inside `ln π` of the E-step.
const MIN_WEIGHT_LOG: f32 = 1e-12;
/// Smallest responsibility mass a component's statistics are divided by.
const MIN_COMPONENT_MASS: f32 = 1e-8;
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
    /// EM stops when the mean per-spike log-likelihood changes by less than this (sklearn `tol`).
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
            tolerance: DEFAULT_TOLERANCE,
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

    /// Fits GMM for a fixed cluster count `k` (EM on `client`'s device).
    pub fn fit_k(
        &self,
        client: &Client,
        features: &[f32],
        num_spikes: usize,
        num_features: usize,
        k: usize,
        feature_mask: Option<&[f32]>,
    ) -> GmmResult {
        fit_gmm_single_k(
            client,
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
        client: &Client,
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
            let candidate = self.fit_k(client, features, num_spikes, num_features, k, feature_mask);
            if best.as_ref().map_or(true, |b| candidate.bic < b.bic) {
                best = Some(candidate);
            }
        }
        best.unwrap()
    }
}

/// Convenience function to fit a GMM with automatic BIC selection over `k_min..=k_max`.
pub fn cluster_gmm_bic(
    client: &Client,
    features: &[f32],
    num_spikes: usize,
    num_features: usize,
    k_min: usize,
    k_max: usize,
) -> GmmResult {
    GmmClusterer::new(k_min, k_max, GmmCovarianceKind::Full).fit(client, features, num_spikes, num_features, None)
}

#[allow(clippy::too_many_arguments)]
fn fit_gmm_single_k(
    client: &Client,
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
    let reg = reg.max(MIN_REGULARIZATION);

    // 1. Deterministic farthest-first seeding + Lloyd steps
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
        weights[c] = (cnt / (n as f32)).max(MIN_INIT_WEIGHT);
        let cov_c = &mut covariances[c * d * d..(c + 1) * d * d];
        for elem in cov_c.iter_mut() {
            *elem /= cnt;
        }
        for f in 0..d {
            // Same ridge as every M-step (was regularized twice: `max(reg) + reg`)
            cov_c[f * d + f] += reg;
        }
    }

    // 2. EM on the device: features, mask and responsibilities stay there; per iteration only the
    //    component parameters go up and their sums come back
    let ln_2pi_d = (d as f64) * (2.0 * std::f64::consts::PI).ln();
    let masked = cov_kind == GmmCovarianceKind::Masked && feature_mask.is_some();
    let features_h = buffer::upload(client, features);
    let mask_h = match feature_mask.filter(|_| masked) {
        Some(m) => buffer::upload(client, m),
        None => buffer::empty::<f32>(client, 1),
    };
    let mask_len = if masked { n * d } else { 1 };
    let resp_h = buffer::empty::<f32>(client, n * k);
    let log_lik_h = buffer::empty::<f32>(client, n);
    let mahal_h = buffer::empty::<f32>(client, n);
    let (ll_mean_h, ll_std_h) = (buffer::empty::<f32>(client, 1), buffer::empty::<f32>(client, 1));
    let per_spike = LaunchGeometry::elementwise(client, n);

    // E-step with the current parameters; returns the mean per-spike log-likelihood
    let e_step = |means: &[f32], covariances: &[f32], weights: &[f32]| -> f64 {
        let mut precisions = vec![0.0f32; k * d * d];
        let mut log_norms = vec![0.0f32; k];
        for c in 0..k {
            let cov_c = &covariances[c * d * d..(c + 1) * d * d];
            let (prec, log_det) = if cov_kind == GmmCovarianceKind::Diagonal {
                diagonal_inverse_logdet(cov_c, d, reg)
            } else {
                invert_spd_and_logdet(cov_c, d, reg)
            };
            precisions[c * d * d..(c + 1) * d * d].copy_from_slice(&prec);
            log_norms[c] = ((weights[c].max(MIN_WEIGHT_LOG) as f64).ln() - 0.5 * (ln_2pi_d + log_det)) as f32;
        }
        // SAFETY: every array is passed with the length it was created with
        unsafe {
            gmm_e_step_kernel::launch::<f32>(
                client,
                per_spike.cube_count.clone(),
                per_spike.cube_dim.clone(),
                BufferArg::from_raw_parts(features_h.clone(), n * d),
                BufferArg::from_raw_parts(buffer::upload(client, means), k * d),
                BufferArg::from_raw_parts(buffer::upload(client, &precisions), k * d * d),
                BufferArg::from_raw_parts(buffer::upload(client, &log_norms), k),
                BufferArg::from_raw_parts(resp_h.clone(), n * k),
                BufferArg::from_raw_parts(log_lik_h.clone(), n),
                BufferArg::from_raw_parts(mahal_h.clone(), n),
                n as u32,
                d as u32,
                k as u32,
            );
        }
        reduce::row_mean_std::<f32>(client, &log_lik_h, &ll_mean_h, &ll_std_h, 1, n);
        buffer::download::<f32>(client, ll_mean_h.clone())[0] as f64
    };

    let mut prev_mean_ll = f64::NEG_INFINITY;
    let mut mean_ll = f64::NEG_INFINITY;
    for _iter in 0..max_iters.max(1) {
        mean_ll = e_step(&means, &covariances, &weights);
        // Converged when the mean per-spike log-likelihood changes by less than `tol` (sklearn)
        if (mean_ll - prev_mean_ll).abs() < tol {
            break;
        }
        prev_mean_ll = mean_ll;

        // M-step: weighted sums on the device, normalization on the host
        let sums_h = buffer::empty::<f32>(client, k * d);
        let mask_sums_h = buffer::empty::<f32>(client, k * d);
        let resp_sums_h = buffer::empty::<f32>(client, k);
        let per_feature = LaunchGeometry::elementwise(client, k * d);
        // SAFETY: as above
        unsafe {
            gmm_mean_sums_kernel::launch::<f32>(
                client,
                per_feature.cube_count,
                per_feature.cube_dim,
                BufferArg::from_raw_parts(features_h.clone(), n * d),
                BufferArg::from_raw_parts(resp_h.clone(), n * k),
                BufferArg::from_raw_parts(mask_h.clone(), mask_len),
                BufferArg::from_raw_parts(sums_h.clone(), k * d),
                BufferArg::from_raw_parts(mask_sums_h.clone(), k * d),
                BufferArg::from_raw_parts(resp_sums_h.clone(), k),
                n as u32,
                d as u32,
                k as u32,
                u32::from(masked),
            );
        }
        let sums = buffer::download::<f32>(client, sums_h);
        let mask_sums = buffer::download::<f32>(client, mask_sums_h);
        let nk = buffer::download::<f32>(client, resp_sums_h);
        for c in 0..k {
            let nk_safe = nk[c].max(MIN_COMPONENT_MASS);
            weights[c] = (nk[c] / n as f32).clamp(MIN_WEIGHT, 1.0);
            for f in 0..d {
                means[c * d + f] = sums[c * d + f] / nk_safe;
            }
        }

        let scatter_h = buffer::empty::<f32>(client, k * d * d);
        let per_entry = LaunchGeometry::elementwise(client, k * d * d);
        // SAFETY: as above
        unsafe {
            gmm_scatter_kernel::launch::<f32>(
                client,
                per_entry.cube_count,
                per_entry.cube_dim,
                BufferArg::from_raw_parts(features_h.clone(), n * d),
                BufferArg::from_raw_parts(resp_h.clone(), n * k),
                BufferArg::from_raw_parts(buffer::upload(client, &means), k * d),
                BufferArg::from_raw_parts(scatter_h.clone(), k * d * d),
                n as u32,
                d as u32,
                k as u32,
                u32::from(cov_kind == GmmCovarianceKind::Diagonal),
            );
        }
        let scatter = buffer::download::<f32>(client, scatter_h);
        for c in 0..k {
            let nk_safe = nk[c].max(MIN_COMPONENT_MASS);
            let cov_c = &mut covariances[c * d * d..(c + 1) * d * d];
            for (v, s) in cov_c.iter_mut().zip(&scatter[c * d * d..(c + 1) * d * d]) {
                *v = s / nk_safe;
            }
            for f in 0..d {
                if masked {
                    // Masked EM: unobserved share of the feature shrinks toward the unit noise prior
                    let obs_frac = (mask_sums[c * d + f] / nk_safe).clamp(0.0, 1.0);
                    cov_c[f * d + f] = obs_frac * cov_c[f * d + f] + (1.0 - obs_frac) + reg;
                } else {
                    cov_c[f * d + f] += reg;
                }
            }
        }
    }

    // 3. Final responsibilities, labels and Mahalanobis distances from the final parameters
    let final_ll = e_step(&means, &covariances, &weights);
    if final_ll.is_finite() {
        mean_ll = final_ll;
    }
    let log_likelihood = mean_ll * n as f64;
    let responsibilities = buffer::download::<f32>(client, resp_h.clone());
    let mahalanobis_sq = buffer::download::<f32>(client, mahal_h.clone());
    let labels: Vec<i32> = (0..n)
        .map(|i| {
            let row = &responsibilities[i * k..(i + 1) * k];
            let best = (0..k).fold(0, |b, c| if row[c] > row[b] { c } else { b });
            match outlier_mahal_sq {
                Some(max_d2) if mahalanobis_sq[i] > max_d2 => -1,
                _ => best as i32,
            }
        })
        .collect();

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
    // First centroid: the point farthest from the overall mean; then farthest-first, then Lloyd steps
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

    // Lloyd k-means iterations settle the seeds in high-density cluster cores
    for _ in 0..SEED_LLOYD_ITERATIONS {
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

/// Precision and `ln det` of a diagonal covariance (each variance at least `reg`).
fn diagonal_inverse_logdet(cov: &[f32], d: usize, reg: f32) -> (Vec<f32>, f64) {
    let mut inv = vec![0.0f32; d * d];
    let mut log_det = 0.0f64;
    for f in 0..d {
        let v = cov[f * d + f].max(reg);
        inv[f * d + f] = 1.0 / v;
        log_det += (v as f64).ln();
    }
    (inv, log_det)
}

/// Times the diagonal jitter grows (×10 each, from `reg`) before giving up on a covariance that
/// is not positive definite.
const CHOLESKY_JITTER_ATTEMPTS: usize = 6;

/// Precision matrix and `ln det` of covariance `cov` (`[d, d]`) by Cholesky (exact). A covariance
/// that is not positive definite gets `reg`, `10·reg`, … added to its diagonal until it is; if
/// none works, the diagonal alone (each variance at least `reg`) is used.
fn invert_spd_and_logdet(cov: &[f32], d: usize, reg: f32) -> (Vec<f32>, f64) {
    let mut a: Vec<f64> = cov.iter().map(|&v| v as f64).collect();
    let mut jitter = 0.0f64;
    for attempt in 0..=CHOLESKY_JITTER_ATTEMPTS {
        if let Some((inv, log_det)) = spd_inverse_logdet(&a, d) {
            return (inv.into_iter().map(|v| v as f32).collect(), log_det);
        }
        let next = reg as f64 * 10f64.powi(attempt as i32);
        for f in 0..d {
            a[f * d + f] += next - jitter;
        }
        jitter = next;
    }
    let mut inv = vec![0.0f32; d * d];
    let mut log_det = 0.0f64;
    for f in 0..d {
        let v = cov[f * d + f].max(reg);
        inv[f * d + f] = 1.0 / v;
        log_det += (v as f64).ln();
    }
    (inv, log_det)
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

        struct Task<'a>(&'a [f32], usize, usize);
        impl dsp_core::compute::ComputeTask for Task<'_> {
            type Output = GmmResult;
            fn run(self, client: Client) -> GmmResult {
                cluster_gmm_bic(&client, self.0, self.1, self.2, 1, 6)
            }
        }
        let targets = dsp_core::compute::ComputeTarget::available();
        assert!(!targets.is_empty(), "no CubeCL runtime compiled in");
        for target in targets {
        let res = target.run(Task(&features, n, d)).expect("runtime");
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
}
