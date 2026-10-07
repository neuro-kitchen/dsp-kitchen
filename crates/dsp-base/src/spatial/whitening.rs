use cubecl::prelude::*;

use crate::core::{buffer, cast, to_f64, DspFloat};
use crate::linalg::{covariance_of_host, symmetric_eigen, symmetric_eigen_batched, EigenOptions, SymmetricEigen};

/// Floor of the whitening regularization `ε` added to every eigenvalue.
pub const MIN_WHITENING_EPSILON: f32 = 1e-12;

/// Spatial whitening transformation across an electrode array (`[channels, channels]` row-major).
///
/// Supports both:
/// - **Global ZCA (Mahalanobis) Whitening**: $\mathbf{W}_{\text{ZCA}} = \mathbf{U} (\mathbf{\Lambda} + \epsilon \mathbf{I})^{-1/2} \mathbf{U}^\top$
/// - **Local $K$-NN Whitening (Kilosort4 style)**: Computes a local ZCA whitening row for each
///   channel $c$ using only its $K$ nearest spatial neighbors, preserving local spatial geometry
///   on high-density Neuropixels and HD-EMG arrays while avoiding overfitting distant noise.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialWhitening {
    pub num_channels: usize,
    /// Row-major `[num_channels, num_channels]` whitening matrix $\mathbf{W}$.
    pub matrix: Vec<f32>,
}

impl SpatialWhitening {
    /// Creates a `SpatialWhitening` transform from a precomputed `[channels, channels]` row-major matrix.
    pub fn from_matrix(num_channels: usize, matrix: Vec<f32>) -> Self {
        assert_eq!(matrix.len(), num_channels * num_channels);
        Self { num_channels, matrix }
    }

    /// Fits a global ZCA whitening matrix on host `data` (`[channels, samples]`): covariance and
    /// eigendecomposition on the device in `F`.
    ///
    /// Regularization `epsilon` (at least [`MIN_WHITENING_EPSILON`]) prevents blow-up on rank-deficient
    /// or flat channels.
    pub fn fit_zca<F: DspFloat>(client: &Client, data: &[f32], channels: usize, samples: usize, epsilon: f32) -> Self {
        assert!(channels > 0 && samples > 0);
        let (cov, _) = covariance_of_host::<F>(client, data, channels, samples);
        let eig = symmetric_eigen::<F>(client, &cov, channels, EigenOptions::default());
        let matrix = zca_from_eigen(&eig, epsilon.max(MIN_WHITENING_EPSILON) as f64);
        Self { num_channels: channels, matrix: matrix.iter().map(|&w| w as f32).collect() }
    }

    /// Fits a local `K`-nearest-neighbour ZCA whitening matrix (Kilosort4 style) given sensor
    /// `positions` (`(x, y)` per channel) and host `data` (`[channels, samples]`).
    ///
    /// For each channel `c`, the `k_neighbors` closest channels (including `c`) give a `[K, K]` local
    /// covariance whose ZCA row for `c` is scattered into row `c` of the `[C, C]` matrix. The full
    /// covariance is computed once on the device and every local eigendecomposition runs in one
    /// batched solve.
    pub fn fit_local_knn<F: DspFloat>(
        client: &Client,
        data: &[f32],
        channels: usize,
        samples: usize,
        positions: &[[f32; 2]],
        k_neighbors: usize,
        epsilon: f32,
    ) -> Self {
        let (cov, _) = covariance_of_host::<F>(client, data, channels, samples);
        let full_cov: Vec<f64> = buffer::download::<F>(client, cov).into_iter().map(to_f64).collect();
        Self::local_knn_from_covariance::<F>(client, &full_cov, channels, positions, k_neighbors, epsilon)
    }

    /// Local ZCA whitening from a `[channels, channels]` covariance (row-major), for callers that
    /// accumulate the covariance over many chunks: each channel is whitened over its
    /// `k_neighbors` nearest contacts (`positions`), all neighbourhoods eigendecomposed in one
    /// batch on `client`.
    pub fn local_knn_from_covariance<F: DspFloat>(
        client: &Client,
        full_cov: &[f64],
        channels: usize,
        positions: &[[f32; 2]],
        k_neighbors: usize,
        epsilon: f32,
    ) -> Self {
        assert_eq!(positions.len(), channels, "Positions length must equal channels");
        assert_eq!(full_cov.len(), channels * channels, "covariance must be [channels, channels]");
        let k = k_neighbors.clamp(1, channels);
        let eps = epsilon.max(MIN_WHITENING_EPSILON) as f64;

        let neighbourhoods: Vec<Vec<usize>> = (0..channels)
            .map(|c| {
                let mut by_distance: Vec<(usize, f32)> = (0..channels)
                    .map(|j| {
                        let (dx, dy) = (positions[c][0] - positions[j][0], positions[c][1] - positions[j][1]);
                        (j, dx * dx + dy * dy)
                    })
                    .collect();
                by_distance.sort_by(|a, b| a.1.total_cmp(&b.1));
                by_distance.into_iter().take(k).map(|(j, _)| j).collect()
            })
            .collect();
        let local: Vec<F> = neighbourhoods
            .iter()
            .flat_map(|nb| nb.iter().flat_map(|&gi| nb.iter().map(move |&gj| (gi, gj))).collect::<Vec<_>>())
            .map(|(gi, gj)| cast::<F>(full_cov[gi * channels + gj]))
            .collect();
        let local = buffer::upload(client, &local);
        let eigs = symmetric_eigen_batched::<F>(client, &local, channels, k, EigenOptions::default());

        let mut matrix = vec![0.0f32; channels * channels];
        for (c, (nb, eig)) in neighbourhoods.iter().zip(&eigs).enumerate() {
            let zca = zca_from_eigen(eig, eps);
            let own = nb.iter().position(|&j| j == c).unwrap_or(0);
            for (lj, &gj) in nb.iter().enumerate() {
                matrix[c * channels + gj] = zca[own * k + lj] as f32;
            }
        }
        Self { num_channels: channels, matrix }
    }

    /// Applies the spatial whitening matrix `[C, C]` to `data` (`[C, S]`) on the CPU.
    pub fn apply_cpu(&self, data: &[f32], channels: usize, samples: usize) -> Vec<f32> {
        assert_eq!(channels, self.num_channels);
        assert_eq!(data.len(), channels * samples);
        let mut out = vec![0.0f32; channels * samples];

        for out_ch in 0..channels {
            let w_row = &self.matrix[out_ch * channels..(out_ch + 1) * channels];
            let out_row = &mut out[out_ch * samples..(out_ch + 1) * samples];
            for in_ch in 0..channels {
                let w = w_row[in_ch];
                if w.abs() > 0.0 {
                    let in_row = &data[in_ch * samples..(in_ch + 1) * samples];
                    for t in 0..samples {
                        out_row[t] += w * in_row[t];
                    }
                }
            }
        }
        out
    }

    /// The operator uploaded once as `F` (dense or sparse rows), for repeated calls.
    pub fn to_device<F: DspFloat>(&self, client: &Client) -> super::DeviceSpatialMatrix {
        super::DeviceSpatialMatrix::upload::<F>(client, &self.matrix, self.num_channels)
    }

    /// One-off: applies the whitening matrix `[C, C]` to `input` (`[C, S]`) on the device (uploads the
    /// matrix; use [`Self::to_device`] for repeated calls).
    pub fn apply_gpu<F: DspFloat>(
        &self,
        client: &Client,
        input: &cubecl::server::Handle,
        output: &cubecl::server::Handle,
        channels: usize,
        samples: usize,
    ) {
        assert_eq!(channels, self.num_channels);
        self.to_device::<F>(client).apply::<F>(client, input, output, channels, samples);
    }
}

/// `W_ZCA = U · diag(1 / √(λ + ε)) · Uᵀ` from an eigendecomposition (negative eigenvalues clamp to 0).
fn zca_from_eigen(eig: &SymmetricEigen, epsilon: f64) -> Vec<f64> {
    let n = eig.n;
    let inv: Vec<f64> = eig.values.iter().map(|&lam| 1.0 / (lam.max(0.0) + epsilon).sqrt()).collect();
    let mut w = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let sum: f64 = (0..n).map(|k| eig.vectors[i * n + k] * inv[k] * eig.vectors[j * n + k]).sum();
            w[i * n + j] = sum;
            w[j * n + i] = sum;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host covariance (`/ samples`), the reference for the whitened output.
    fn host_covariance(data: &[f32], channels: usize, samples: usize) -> Vec<f64> {
        let means: Vec<f64> = (0..channels).map(|c| data[c * samples..(c + 1) * samples].iter().map(|v| *v as f64).sum::<f64>() / samples as f64).collect();
        (0..channels * channels)
            .map(|e| {
                let (i, j) = (e / channels, e % channels);
                (0..samples).map(|t| (data[i * samples + t] as f64 - means[i]) * (data[j * samples + t] as f64 - means[j])).sum::<f64>() / samples as f64
            })
            .collect()
    }

    fn zca_whitens(client: &Client) {
        let channels = 4;
        let samples = 2000;
        let mut z = vec![0.0f32; channels * samples];
        let mut state = 0x9876_5432u64;
        let mut next_norm = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((state >> 32) as f32 / (u32::MAX as f32)).clamp(1e-6, 1.0 - 1e-6);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (state >> 32) as f32 / (u32::MAX as f32);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        };

        for v in &mut z {
            *v = next_norm();
        }

        // Mix independent normal channels with a strongly correlated mixing matrix A
        let a: [f32; 16] = [
            2.5, 1.2, 0.4, 0.1,
            1.2, 3.0, 0.8, 0.2,
            0.4, 0.8, 1.8, 0.6,
            0.1, 0.2, 0.6, 2.2,
        ];
        let mut x = vec![0.0f32; channels * samples];
        for i in 0..channels {
            for k in 0..channels {
                let w = a[i * channels + k];
                for t in 0..samples {
                    x[i * samples + t] += w * z[k * samples + t];
                }
            }
        }

        let whiten = SpatialWhitening::fit_zca::<f32>(client, &x, channels, samples, 1e-5);
        let y_cpu = whiten.apply_cpu(&x, channels, samples);

        // Verify sample covariance of whitened signal is identity I_4
        let cov_y = host_covariance(&y_cpu, channels, samples);
        for i in 0..channels {
            for j in 0..channels {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (cov_y[i * channels + j] - target).abs() < 1e-2,
                    "cov_y[{i},{j}] = {}, expected {target}",
                    cov_y[i * channels + j]
                );
            }
        }

        // The device projection matches the host product
        let input = buffer::upload(client, &x);
        let output = buffer::empty::<f32>(client, x.len());
        whiten.apply_gpu::<f32>(client, &input, &output, channels, samples);
        let y_dev = buffer::download::<f32>(client, output);
        for (a, b) in y_cpu.iter().zip(&y_dev) {
            assert!((a - b).abs() < 1e-4, "{}: {a} vs {b}", client.name());
        }
    }
    runtime_test!(test_zca_whitening_produces_identity_covariance, zca_whitens);
}
