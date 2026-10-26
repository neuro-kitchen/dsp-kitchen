use cubecl::prelude::*;
use crate::core::DspFloat;
use super::covariance::covariance_of_host;
use super::eigen::{symmetric_eigen, EigenOptions};
use super::projection::DeviceProjection;

/// Floor of the isotropic noise variance (keeps `M = WᵀW + σ²I` invertible).
pub const MIN_NOISE_VARIANCE: f64 = 1e-8;

/// Noise variance used when every component is kept (no discarded eigenvalues to average).
pub const FULL_RANK_NOISE_VARIANCE: f64 = 1e-6;

/// Probabilistic Principal Component Analysis (PPCA) model (Tipping & Bishop, 1999).
///
/// Generative Gaussian latent variable model:
/// $$\mathbf{x} = \mathbf{W}\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_q), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_D)$$
#[derive(Debug, Clone)]
pub struct PpcaModel {
    pub num_channels: usize,
    pub num_components: usize,
    pub mean: Vec<f32>,
    /// Factor loading matrix $\mathbf{W}$ of shape `[num_channels, num_components]` (row-major).
    pub weights: Vec<f32>,
    /// Posterior projection matrix $\mathbf{P} = \mathbf{W} (\mathbf{W}^\top \mathbf{W} + \sigma^2 \mathbf{I})^{-1}$
    /// of shape `[num_channels, num_components]` so $\mathbb{E}[\mathbf{z} \mid \mathbf{x}] = \mathbf{P}^\top (\mathbf{x} - \boldsymbol{\mu})$.
    pub posterior_projection: Vec<f32>,
    /// Isotropic observation noise variance $\sigma^2$.
    pub noise_variance: f32,
    /// Top-$q$ eigenvalues of the sample covariance matrix.
    pub eigenvalues: Vec<f32>,
}

impl PpcaModel {
    /// Fits PPCA in closed form (maximum likelihood from the covariance eigendecomposition) on host
    /// data `[channels, samples]`; covariance and eigendecomposition on the device in `F`.
    pub fn fit<F: DspFloat>(client: &Client, data: &[f32], channels: usize, samples: usize, num_components: usize) -> Self {
        assert!(channels > 0 && samples > 0, "Channels and samples must be > 0");
        let num_components = num_components.clamp(1, channels);
        let (cov, mean) = covariance_of_host::<F>(client, data, channels, samples);
        let eig = symmetric_eigen::<F>(client, &cov, channels, EigenOptions::default());

        // Tipping & Bishop (1999) Eq. 8: sigma_ML^2 is the average of the discarded eigenvalues
        let noise_variance = if num_components < channels {
            let tail: f64 = eig.values[num_components..].iter().map(|&v| v.max(0.0)).sum();
            (tail / (channels - num_components) as f64).max(MIN_NOISE_VARIANCE)
        } else {
            FULL_RANK_NOISE_VARIANCE
        };

        let mut weights = vec![0.0f32; channels * num_components];
        let mut posterior_projection = vec![0.0f32; channels * num_components];
        let mut eigenvalues = Vec::with_capacity(num_components);
        for k in 0..num_components {
            let lam = eig.values[k].max(noise_variance + MIN_NOISE_VARIANCE);
            eigenvalues.push(lam as f32);
            let scale_w = (lam - noise_variance).max(0.0).sqrt();
            // W^T W + sigma^2 I = diag(lam_1, ..., lam_q), so P = W M^-1 scales column k by scale_w / lam
            let scale_p = scale_w / lam;
            for c in 0..channels {
                let u_ck = eig.vectors[c * channels + k];
                weights[c * num_components + k] = (u_ck * scale_w) as f32;
                posterior_projection[c * num_components + k] = (u_ck * scale_p) as f32;
            }
        }

        Self {
            num_channels: channels,
            num_components,
            mean: mean.iter().map(|&m| m as f32).collect(),
            weights,
            posterior_projection,
            noise_variance: noise_variance as f32,
            eigenvalues,
        }
    }

    /// Fits PPCA via Expectation-Maximization (EM) with an observation mask (`[channels, samples]`,
    /// `true` = observed, `false` = missing/outside local neighborhood).
    pub fn fit_em_masked<F: DspFloat>(
        client: &Client,
        data: &[f32],
        observed_mask: &[bool],
        channels: usize,
        samples: usize,
        mut num_components: usize,
        max_iters: usize,
    ) -> Self {
        assert_eq!(data.len(), channels * samples);
        assert_eq!(observed_mask.len(), channels * samples);
        num_components = num_components.clamp(1, channels);

        // Compute observed per-channel means and impute initial matrix for warm-start
        let mut mean = vec![0.0f32; channels];
        for c in 0..channels {
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for t in 0..samples {
                if observed_mask[c * samples + t] {
                    sum += data[c * samples + t];
                    count += 1;
                }
            }
            mean[c] = if count > 0 { sum / (count as f32) } else { 0.0 };
        }

        let mut imputed = data.to_vec();
        for c in 0..channels {
            for t in 0..samples {
                if !observed_mask[c * samples + t] {
                    imputed[c * samples + t] = mean[c];
                }
            }
        }

        let mut model = Self::fit::<F>(client, &imputed, channels, samples, num_components);
        for _iter in 0..max_iters.max(1) {
            // E-step: project current imputed data to posterior latent expectations Z = E[z | x]
            let z = model.project_cpu(&imputed, channels, samples);
            // Reconstruct missing entries x_hat = W * z + mu
            let recon = model.reconstruct_cpu(&z, samples);
            for c in 0..channels {
                for t in 0..samples {
                    let idx = c * samples + t;
                    if !observed_mask[idx] {
                        imputed[idx] = recon[idx];
                    }
                }
            }
            // M-step: re-estimate closed-form subspace on completed sufficient statistics
            model = Self::fit::<F>(client, &imputed, channels, samples, num_components);
        }
        model
    }

    /// Computes posterior latent expectations $\mathbb{E}[\mathbf{z} \mid \mathbf{x}] = \mathbf{M}^{-1}\mathbf{W}^\top(\mathbf{x} - \boldsymbol{\mu})$
    /// of shape `[num_components, samples]` on the CPU.
    pub fn project_cpu(&self, data: &[f32], channels: usize, samples: usize) -> Vec<f32> {
        assert_eq!(channels, self.num_channels);
        assert_eq!(data.len(), channels * samples);
        let mut out = vec![0.0f32; self.num_components * samples];

        for k in 0..self.num_components {
            let out_row = &mut out[k * samples..(k + 1) * samples];
            for c in 0..channels {
                let p_ck = self.posterior_projection[c * self.num_components + k];
                let mean_c = self.mean[c];
                let in_row = &data[c * samples..(c + 1) * samples];
                for t in 0..samples {
                    out_row[t] += (in_row[t] - mean_c) * p_ck;
                }
            }
        }
        out
    }

    /// Reconstructs observations $\hat{\mathbf{x}} = \mathbf{W}\mathbf{z} + \boldsymbol{\mu}$ (`[channels, samples]`) from latent codes `z` (`[num_components, samples]`).
    pub fn reconstruct_cpu(&self, z: &[f32], samples: usize) -> Vec<f32> {
        assert_eq!(z.len(), self.num_components * samples);
        let mut recon = vec![0.0f32; self.num_channels * samples];

        for c in 0..self.num_channels {
            let out_row = &mut recon[c * samples..(c + 1) * samples];
            out_row.fill(self.mean[c]);
            for k in 0..self.num_components {
                let w_ck = self.weights[c * self.num_components + k];
                let z_row = &z[k * samples..(k + 1) * samples];
                for t in 0..samples {
                    out_row[t] += w_ck * z_row[t];
                }
            }
        }
        recon
    }

    /// The projection with its weights uploaded once as `F`, for repeated calls.
    pub fn to_device<F: DspFloat>(&self, client: &Client) -> DeviceProjection {
        DeviceProjection::upload::<F>(client, &self.posterior_projection, &self.mean, self.num_channels, self.num_components)
    }

    /// One-off device projection of `[channels, samples]` into `[num_components, samples]` (uploads
    /// the weights; use [`Self::to_device`] for repeated calls).
    pub fn project_gpu<F: DspFloat>(
        &self,
        client: &Client,
        input_handle: &cubecl::server::Handle,
        output_handle: &cubecl::server::Handle,
        channels: usize,
        samples: usize,
    ) {
        assert_eq!(channels, self.num_channels);
        self.to_device::<F>(client).project::<F>(client, input_handle, output_handle, samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recovers(client: &Client) {
        let channels = 5;
        let samples = 3000;
        let true_sigma2 = 0.25f32; // sigma = 0.5
        let true_sigma = true_sigma2.sqrt();

        let mut state = 0xCAFE_BABEu64;
        let mut next_norm = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((state >> 32) as f32 / (u32::MAX as f32)).clamp(1e-6, 1.0 - 1e-6);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (state >> 32) as f32 / (u32::MAX as f32);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        };

        // 1D latent factor z ~ N(0, 1) with loading w = [3, -2, 1, 0, 0]^T + isotropic noise
        let w_true = [3.0f32, -2.0, 1.0, 0.0, 0.0];
        let mut data = vec![0.0f32; channels * samples];
        for t in 0..samples {
            let z = next_norm();
            for c in 0..channels {
                data[c * samples + t] = w_true[c] * z + true_sigma * next_norm();
            }
        }

        let ppca = PpcaModel::fit::<f32>(client, &data, channels, samples, 1);
        assert!((ppca.noise_variance - true_sigma2).abs() < 0.05, "noise_var={}", ppca.noise_variance);

        // Norm of W_ML should be close to ||w_true|| = sqrt(9 + 4 + 1) = sqrt(14) = 3.7417
        let w_norm_sq: f32 = ppca.weights.iter().map(|&v| v * v).sum();
        assert!((w_norm_sq - 14.0).abs() < 1.0, "w_norm_sq={w_norm_sq}");
    }
    runtime_test!(test_ppca_recovers_subspace_and_isotropic_noise_variance, recovers);
}
