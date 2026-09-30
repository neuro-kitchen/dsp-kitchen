//! `ConvAutoencoderEmbedder`: 1D Convolutional Autoencoder learning non-linear
//! low-dimensional representations of multi-channel waveforms `[N, K, T] -> [N, D]`.
//!
//! Implements `dsp_synapse::traits::FeatureEmbedder`.

use anyhow::Result;
use dsp_synapse::{FeatureEmbedder, SnippetBatch};
use crate::backbones::{LinearLayer, ResBlock1D};
use crate::backend::{SynapseMlDevice, Tensor2D, Tensor3D, snippet_batch_to_tensor};
use crate::hub::SafetensorsMap;

#[derive(Debug, Clone)]
pub struct ConvAutoencoderEmbedder {
    pub enc_block1: ResBlock1D,
    pub enc_block2: ResBlock1D,
    pub latent_proj: LinearLayer,
    pub dec_proj: LinearLayer,
    pub k_channels: usize,
    pub snippet_samples: usize,
    pub latent_dim: usize,
    pub device: SynapseMlDevice,
}

impl ConvAutoencoderEmbedder {
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
            latent_proj: LinearLayer::new_initialized(32, d, seed.wrapping_add(200), device),
            dec_proj: LinearLayer::new_initialized(d, k * t, seed.wrapping_add(300), device),
            k_channels: k,
            snippet_samples: t,
            latent_dim: d,
            device,
        }
    }

    /// Encodes a 3D batch `[N, K, T]` into a 2D latent matrix `[N, latent_dim]`.
    pub fn encode_tensor(&self, input: &Tensor3D) -> Tensor2D {
        let h1 = self.enc_block1.forward(input).max_pool1d(2, 2);
        let h2 = self.enc_block2.forward(&h1).global_avg_pool_1d();
        self.latent_proj.forward(&h2)
    }

    /// Decodes a 2D latent matrix `[N, latent_dim]` back into a 3D waveform batch `[N, K, T]`.
    pub fn decode_tensor(&self, z: &Tensor2D) -> Tensor3D {
        let flat = self.dec_proj.forward(z);
        flat.unflatten_channels_time(self.k_channels, self.snippet_samples)
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.enc_block1.save_weights("conv_ae.enc_block1", map);
        self.enc_block2.save_weights("conv_ae.enc_block2", map);
        self.latent_proj.save_weights("conv_ae.latent_proj", map);
        self.dec_proj.save_weights("conv_ae.dec_proj", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.enc_block1
            .load_weights("conv_ae.enc_block1", map, self.device)?;
        self.enc_block2
            .load_weights("conv_ae.enc_block2", map, self.device)?;
        self.latent_proj
            .load_weights("conv_ae.latent_proj", map, self.device)?;
        self.dec_proj
            .load_weights("conv_ae.dec_proj", map, self.device)?;
        Ok(())
    }
}

impl FeatureEmbedder for ConvAutoencoderEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> dsp_core::DspResult<(Vec<f32>, usize)> {
        if batch.num_spikes == 0 {
            return Ok((Vec::new(), self.latent_dim));
        }
        let x = snippet_batch_to_tensor(batch, self.device).mul_scalar(0.01);
        let z = self.encode_tensor(&x);
        Ok((z.into_vec(), self.latent_dim))
    }
}
