use cubecl::prelude::*;
use super::svd::SymmetricEig;
use super::kernels::pca_project_kernel;
use dsp_core::compute::LaunchGeometry;

/// Fitted Principal Component Analysis model.
#[derive(Debug, Clone)]
pub struct PcaModel {
    pub num_channels: usize,
    pub num_components: usize,
    pub mean: Vec<f32>,
    /// Matrix of shape [channels, components] in row-major order.
    pub components: Vec<f32>,
    pub explained_variance: Vec<f32>,
    pub explained_variance_ratio: Vec<f32>,
}

impl PcaModel {
    /// Fits PCA on a data matrix of shape [channels, samples].
    pub fn fit(data: &[f32], channels: usize, samples: usize, mut num_components: usize) -> Self {
        assert_eq!(data.len(), channels * samples, "Data size mismatch");
        assert!(channels > 0 && samples > 0, "Channels and samples must be > 0");

        num_components = num_components.min(channels);

        // 1. Compute channel means
        let mut mean = vec![0.0f32; channels];
        for c in 0..channels {
            let offset = c * samples;
            let sum: f32 = data[offset..offset + samples].iter().sum();
            mean[c] = sum / samples as f32;
        }

        // 2. Compute covariance matrix [channels, channels]
        let mut cov = vec![0.0f32; channels * channels];
        let inv_samples = 1.0f32 / (samples as f32).max(1.0);

        for i in 0..channels {
            let off_i = i * samples;
            let mean_i = mean[i];

            for j in i..channels {
                let off_j = j * samples;
                let mean_j = mean[j];

                let mut dot = 0.0f32;
                for t in 0..samples {
                    let vi = data[off_i + t] - mean_i;
                    let vj = data[off_j + t] - mean_j;
                    dot += vi * vj;
                }
                let val = dot * inv_samples;
                cov[i * channels + j] = val;
                cov[j * channels + i] = val;
            }
        }

        // 3. Eigendecomposition of covariance matrix
        let eig = SymmetricEig::decompose(&cov, channels, 100);

        // 4. Extract top k components
        let total_variance: f32 = eig.eigenvalues.iter().copied().filter(|&v| v > 0.0).sum();
        let total_var_denom = if total_variance > 0.0 { total_variance } else { 1.0 };

        let mut components = vec![0.0f32; channels * num_components];
        let mut explained_variance = Vec::with_capacity(num_components);
        let mut explained_variance_ratio = Vec::with_capacity(num_components);

        for k in 0..num_components {
            let val = eig.eigenvalues[k].max(0.0);
            explained_variance.push(val);
            explained_variance_ratio.push(val / total_var_denom);

            for c in 0..channels {
                components[c * num_components + k] = eig.eigenvectors[c * channels + k];
            }
        }

        Self {
            num_channels: channels,
            num_components,
            mean,
            components,
            explained_variance,
            explained_variance_ratio,
        }
    }

    /// Projects input data [channels, samples] into PCA space [num_components, samples] using CPU.
    pub fn project_cpu(&self, data: &[f32], channels: usize, samples: usize) -> Vec<f32> {
        assert_eq!(channels, self.num_channels);
        let mut output = vec![0.0f32; self.num_components * samples];

        for k in 0..self.num_components {
            let out_offset = k * samples;
            for t in 0..samples {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    let centered = data[c * samples + t] - self.mean[c];
                    let weight = self.components[c * self.num_components + k];
                    sum += centered * weight;
                }
                output[out_offset + t] = sum;
            }
        }

        output
    }

    /// Projects input data [channels, samples] into PCA space [num_components, samples] using CubeCL GPU.
    pub fn project_gpu<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        input_handle: &cubecl::server::Handle,
        output_handle: &cubecl::server::Handle,
        channels: usize,
        samples: usize,
    ) {
        assert_eq!(channels, self.num_channels);

        let comp_bytes = f32::as_bytes(&self.components);
        let mean_bytes = f32::as_bytes(&self.mean);

        let comp_handle = client.create_from_slice(comp_bytes);
        let mean_handle = client.create_from_slice(mean_bytes);

        let geom = LaunchGeometry::channels_samples(client, self.num_components, samples);
        let total_in = channels * samples;
        let total_out = self.num_components * samples;

        unsafe {
            pca_project_kernel::launch::<R>(
                client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(input_handle.clone(), total_in),
                ArrayArg::from_raw_parts(comp_handle, channels * self.num_components),
                ArrayArg::from_raw_parts(mean_handle, channels),
                ArrayArg::from_raw_parts(output_handle.clone(), total_out),
                channels as u32,
                samples as u32,
                self.num_components as u32,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_fit_and_project() {
        // 2 channels, 10 samples with strong correlation
        let ch0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ch1 = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let mut data = Vec::new();
        data.extend_from_slice(&ch0);
        data.extend_from_slice(&ch1);

        let pca = PcaModel::fit(&data, 2, 10, 1);
        assert_eq!(pca.num_components, 1);
        assert!(pca.explained_variance_ratio[0] > 0.99);

        let proj = pca.project_cpu(&data, 2, 10);
        assert_eq!(proj.len(), 10);
    }
}
