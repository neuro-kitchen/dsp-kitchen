//! `DartsortVaeEmbedder`: Variational Autoencoder (Dartsort-inspired) producing
//! probabilistic Gaussian latent embeddings $(\boldsymbol{\mu}, \log \boldsymbol{\sigma}^2)$.
//!
//! Implements `dsp_synapse::traits::FeatureEmbedder`.

use anyhow::Result;
use dsp_synapse::{FeatureEmbedder, SnippetBatch};
use crate::backbones::{LinearLayer, ResBlock1D};
use crate::backend::{SynapseMlDevice, Tensor2D, Tensor3D, snippet_batch_to_tensor};
use crate::hub::SafetensorsMap;

/// Posterior Gaussian latent distribution parameters for a batch of spikes.
#[derive(Debug, Clone)]
pub struct VaePosterior {
    pub mu: Tensor2D,
    pub log_var: Tensor2D,
}

impl VaePosterior {
    /// Analytical Kullback-Leibler divergence $D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I))$ per spike.
    pub fn kl_divergence_per_spike(&self) -> Vec<f32> {
        let [n, d] = self.mu.shape;
        let mut kl = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..d {
                let m = self.mu.data[i * d + j];
                let lv = self.log_var.data[i * d + j].clamp(-10.0, 10.0);
                sum += -0.5 * (1.0 + lv - m * m - lv.exp());
            }
            kl[i] = sum;
        }
        kl
    }
}

#[derive(Debug, Clone)]
pub struct DartsortVaeEmbedder {
    pub enc_block1: ResBlock1D,
    pub enc_block2: ResBlock1D,
    pub fc_mu: LinearLayer,
    pub fc_log_var: LinearLayer,
    pub decoder_fc: LinearLayer,
    pub k_channels: usize,
    pub snippet_samples: usize,
    pub latent_dim: usize,
    pub device: SynapseMlDevice,
}

impl DartsortVaeEmbedder {
    pub fn new(
        k_channels: usize,
        snippet_samples: usize,
        latent_dim: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        let k = k_channels.max(1);
        let t = snippet_samples.max(4);
        let d = latent_dim.max(2);

        Self {
            enc_block1: ResBlock1D::new(k, 16, 5, seed, device),
            enc_block2: ResBlock1D::new(16, 32, 3, seed.wrapping_add(100), device),
            fc_mu: LinearLayer::new_initialized(32, d, seed.wrapping_add(200), device),
            fc_log_var: LinearLayer::new_initialized(32, d, seed.wrapping_add(300), device),
            decoder_fc: LinearLayer::new_initialized(d, k * t, seed.wrapping_add(400), device),
            k_channels: k,
            snippet_samples: t,
            latent_dim: d,
            device,
        }
    }

    pub fn encode_posterior(&self, input: &Tensor3D) -> VaePosterior {
        let h1 = self.enc_block1.forward(input).max_pool1d(2, 2);
        let h2 = self.enc_block2.forward(&h1).global_avg_pool_1d();
        let mu = self.fc_mu.forward(&h2);
        let log_var = self.fc_log_var.forward(&h2);
        VaePosterior { mu, log_var }
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.enc_block1.save_weights("vae.enc_block1", map);
        self.enc_block2.save_weights("vae.enc_block2", map);
        self.fc_mu.save_weights("vae.fc_mu", map);
        self.fc_log_var.save_weights("vae.fc_log_var", map);
        self.decoder_fc.save_weights("vae.decoder_fc", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.enc_block1
            .load_weights("vae.enc_block1", map, self.device)?;
        self.enc_block2
            .load_weights("vae.enc_block2", map, self.device)?;
        self.fc_mu.load_weights("vae.fc_mu", map, self.device)?;
        self.fc_log_var
            .load_weights("vae.fc_log_var", map, self.device)?;
        self.decoder_fc
            .load_weights("vae.decoder_fc", map, self.device)?;
        Ok(())
    }
}

impl FeatureEmbedder for DartsortVaeEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> dsp_core::DspResult<(Vec<f32>, usize)> {
        if batch.num_spikes == 0 {
            return Ok((Vec::new(), self.latent_dim));
        }
        let x = snippet_batch_to_tensor(batch, self.device).mul_scalar(0.01);
        let posterior = self.encode_posterior(&x);
        Ok((posterior.mu.into_vec(), self.latent_dim))
    }
}
