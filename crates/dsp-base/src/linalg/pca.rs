use cubecl::prelude::*;
use super::covariance::covariance_of_host;
use super::eigen::{symmetric_eigen, EigenOptions};
use super::projection::DeviceProjection;
use crate::core::DspFloat;

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
    /// Fits PCA on host data `[channels, samples]`: covariance and eigendecomposition on the device
    /// in `F`.
    pub fn fit<F: DspFloat>(client: &Client, data: &[f32], channels: usize, samples: usize, num_components: usize) -> Self {
        assert!(channels > 0 && samples > 0, "Channels and samples must be > 0");
        let num_components = num_components.min(channels);
        let (cov, mean) = covariance_of_host::<F>(client, data, channels, samples);
        let eig = symmetric_eigen::<F>(client, &cov, channels, EigenOptions::default());

        let total_variance: f64 = eig.values.iter().filter(|&&v| v > 0.0).sum();
        let total = if total_variance > 0.0 { total_variance } else { 1.0 };
        let mut components = vec![0.0f32; channels * num_components];
        let mut explained_variance = Vec::with_capacity(num_components);
        let mut explained_variance_ratio = Vec::with_capacity(num_components);
        for k in 0..num_components {
            let val = eig.values[k].max(0.0);
            explained_variance.push(val as f32);
            explained_variance_ratio.push((val / total) as f32);
            for c in 0..channels {
                components[c * num_components + k] = eig.vectors[c * channels + k] as f32;
            }
        }

        Self {
            num_channels: channels,
            num_components,
            mean: mean.iter().map(|&m| m as f32).collect(),
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

    /// The projection with its weights uploaded once as `F`, for repeated calls.
    pub fn to_device<F: DspFloat>(&self, client: &Client) -> DeviceProjection {
        DeviceProjection::upload::<F>(client, &self.components, &self.mean, self.num_channels, self.num_components)
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

    fn fit_and_project(client: &Client) {
        // 2 channels, 10 samples with strong correlation
        let mut data: Vec<f32> = (1..=10).map(|v| v as f32).collect();
        data.extend((1..=10).map(|v| 2.0 * v as f32));

        let pca = PcaModel::fit::<f32>(client, &data, 2, 10, 1);
        assert_eq!(pca.num_components, 1);
        assert!(pca.explained_variance_ratio[0] > 0.99);
        // The component follows (1, 2) / √5 up to sign
        let (a, b) = (pca.components[0], pca.components[1]);
        assert!((b / a - 2.0).abs() < 1e-3, "{}: component ({a}, {b})", client.name());
        assert_eq!(pca.project_cpu(&data, 2, 10).len(), 10);
    }
    runtime_test!(test_pca_fit_and_project, fit_and_project);
}
