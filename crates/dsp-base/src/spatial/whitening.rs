use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;
use crate::linalg::SymmetricEig;
use super::kernels::spatial_matrix_multiply_kernel;

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

    /// Fits a Global Zero-Phase Component Analysis (ZCA) whitening matrix on `data` (`[channels, samples]`).
    ///
    /// Regularization parameter `epsilon` prevents numerical blow-up on rank-deficient or flat channels.
    pub fn fit_zca(data: &[f32], channels: usize, samples: usize, epsilon: f32) -> Self {
        assert_eq!(data.len(), channels * samples, "Data size mismatch");
        assert!(channels > 0 && samples > 0);

        let cov = compute_sample_covariance(data, channels, samples);
        let matrix = zca_from_covariance(&cov, channels, epsilon.max(1e-12));

        Self {
            num_channels: channels,
            matrix,
        }
    }

    /// Fits a Local $K$-Nearest-Neighbor ZCA whitening matrix (Kilosort4 style) given sensor
    /// `positions` (`[channels, 2]` or `(x, y)` pairs) and `data` (`[channels, samples]`).
    ///
    /// For each channel $c$, selects the `k_neighbors` closest channels (including $c$ itself),
    /// computes the $[K, K]$ local covariance matrix and its ZCA inverse square root, and scatters
    /// the row corresponding to $c$ into row $c$ of the global $[C, C]$ sparse-structured matrix.
    pub fn fit_local_knn(
        data: &[f32],
        channels: usize,
        samples: usize,
        positions: &[[f32; 2]],
        k_neighbors: usize,
        epsilon: f32,
    ) -> Self {
        assert_eq!(data.len(), channels * samples, "Data size mismatch");
        assert_eq!(positions.len(), channels, "Positions length must equal channels");
        let k = k_neighbors.clamp(1, channels);
        let eps = epsilon.max(1e-12);

        let full_cov = compute_sample_covariance(data, channels, samples);
        let mut matrix = vec![0.0f32; channels * channels];

        for c in 0..channels {
            let mut dists: Vec<(usize, f32)> = (0..channels)
                .map(|j| {
                    let dx = positions[c][0] - positions[j][0];
                    let dy = positions[c][1] - positions[j][1];
                    (j, dx * dx + dy * dy)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let neighbors: Vec<usize> = dists.iter().take(k).map(|&(idx, _)| idx).collect();
            let self_local_idx = neighbors.iter().position(|&idx| idx == c).unwrap_or(0);

            let mut local_cov = vec![0.0f32; k * k];
            for (li, &gi) in neighbors.iter().enumerate() {
                for (lj, &gj) in neighbors.iter().enumerate() {
                    local_cov[li * k + lj] = full_cov[gi * channels + gj];
                }
            }

            let local_zca = zca_from_covariance(&local_cov, k, eps);
            for (lj, &gj) in neighbors.iter().enumerate() {
                matrix[c * channels + gj] = local_zca[self_local_idx * k + lj];
            }
        }

        Self {
            num_channels: channels,
            matrix,
        }
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

    /// Applies the spatial whitening matrix `[C, C]` to `input` (`[C, S]`) in VRAM using CubeCL.
    pub fn apply_gpu<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        input: &cubecl::server::Handle,
        output: &cubecl::server::Handle,
        channels: usize,
        samples: usize,
    ) {
        assert_eq!(channels, self.num_channels);
        let weights_handle = client.create_from_slice(f32::as_bytes(&self.matrix));
        execute_spatial_matrix_multiply::<R>(
            client,
            input,
            &weights_handle,
            output,
            channels,
            samples,
        );
    }
}

/// Dispatches the CubeCL `[C, C] x [C, S]` spatial linear projection kernel with pre-uploaded weights.
pub fn execute_spatial_matrix_multiply<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    weights: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    let geom = LaunchGeometry::channels_samples(client, channels, samples);
    let total = channels * samples;

    unsafe {
        spatial_matrix_multiply_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), total),
            ArrayArg::from_raw_parts(weights.clone(), channels * channels),
            ArrayArg::from_raw_parts(output.clone(), total),
            channels as u32,
            samples as u32,
        );
    }
}

fn compute_sample_covariance(data: &[f32], channels: usize, samples: usize) -> Vec<f32> {
    let mut means = vec![0.0f32; channels];
    let inv_s = 1.0 / (samples as f32).max(1.0);
    for c in 0..channels {
        let row = &data[c * samples..(c + 1) * samples];
        means[c] = row.iter().sum::<f32>() * inv_s;
    }

    let mut cov = vec![0.0f32; channels * channels];
    for i in 0..channels {
        let row_i = &data[i * samples..(i + 1) * samples];
        let mi = means[i];
        for j in i..channels {
            let row_j = &data[j * samples..(j + 1) * samples];
            let mj = means[j];
            let mut acc = 0.0f64;
            for t in 0..samples {
                acc += ((row_i[t] - mi) as f64) * ((row_j[t] - mj) as f64);
            }
            let c_ij = (acc * (inv_s as f64)) as f32;
            cov[i * channels + j] = c_ij;
            cov[j * channels + i] = c_ij;
        }
    }
    cov
}

fn zca_from_covariance(cov: &[f32], n: usize, epsilon: f32) -> Vec<f32> {
    let eig = SymmetricEig::decompose(cov, n, 120);
    let inv_scales: Vec<f32> = eig
        .eigenvalues
        .iter()
        .map(|&lam| 1.0 / (lam.max(0.0) + epsilon).sqrt())
        .collect();

    // W_ZCA = U * diag(inv_scales) * U^T
    let mut w = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                let u_ik = eig.eigenvectors[i * n + k];
                let u_jk = eig.eigenvectors[j * n + k];
                sum += u_ik * inv_scales[k] * u_jk;
            }
            w[i * n + j] = sum;
            w[j * n + i] = sum;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn test_zca_whitening_produces_identity_covariance() {
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

        let whiten = SpatialWhitening::fit_zca(&x, channels, samples, 1e-5);
        let y_cpu = whiten.apply_cpu(&x, channels, samples);

        // Verify sample covariance of whitened signal is identity I_4
        let cov_y = compute_sample_covariance(&y_cpu, channels, samples);
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

        // Verify GPU kernel matches CPU
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let in_handle = client.create_from_slice(f32::as_bytes(&x));
        let out_handle = client.empty(channels * samples * 4);
        whiten.apply_gpu::<WgpuRuntime>(&client, &in_handle, &out_handle, channels, samples);
        let y_gpu_bytes = client.read_one_unchecked(out_handle);
        let y_gpu = f32::from_bytes(&y_gpu_bytes);
        for idx in 0..(channels * samples) {
            assert!((y_cpu[idx] - y_gpu[idx]).abs() < 1e-4);
        }
    }
}
