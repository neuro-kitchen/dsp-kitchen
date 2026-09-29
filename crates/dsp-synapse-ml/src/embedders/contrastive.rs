//! `ContrastiveWaveformEmbedder`: SimCLR-inspired metric-learning waveform encoder
//! producing unit-hypersphere $L_2$-normalized embeddings (`||z||_2 = 1.0`).
//!
//! Implements `dsp_synapse::traits::FeatureEmbedder`.

use anyhow::Result;
use dsp_synapse::{FeatureEmbedder, SnippetBatch};
use crate::backbones::{MlpBackbone, ResBlock1D};
use crate::backend::{SynapseMlDevice, Tensor2D, Tensor3D, snippet_batch_to_tensor};
use crate::hub::SafetensorsMap;

#[derive(Debug, Clone)]
pub struct ContrastiveWaveformEmbedder {
    pub res_block1: ResBlock1D,
    pub res_block2: ResBlock1D,
    pub proj_head: MlpBackbone,
    pub latent_dim: usize,
    pub device: SynapseMlDevice,
}

impl ContrastiveWaveformEmbedder {
    pub fn new(
        k_channels: usize,
        latent_dim: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        let d = latent_dim.max(2);
        Self {
            res_block1: ResBlock1D::new(k_channels.max(1), 16, 5, seed, device),
            res_block2: ResBlock1D::new(16, 32, 3, seed.wrapping_add(100), device),
            proj_head: MlpBackbone::new(32, &[32], d, seed.wrapping_add(200), device),
            latent_dim: d,
            device,
        }
    }

    /// Computes $L_2$-normalized projection embeddings `[N, latent_dim]`.
    pub fn encode_normalized(&self, input: &Tensor3D) -> Tensor2D {
        let h1 = self.res_block1.forward(input).max_pool1d(2, 2);
        let h2 = self.res_block2.forward(&h1).global_avg_pool_1d();
        self.proj_head.forward(&h2).l2_normalize(1e-8)
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.res_block1.save_weights("contrastive.res_block1", map);
        self.res_block2.save_weights("contrastive.res_block2", map);
        self.proj_head.save_weights("contrastive.proj_head", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.res_block1
            .load_weights("contrastive.res_block1", map, self.device)?;
        self.res_block2
            .load_weights("contrastive.res_block2", map, self.device)?;
        self.proj_head
            .load_weights("contrastive.proj_head", map, self.device)?;
        Ok(())
    }
}

impl FeatureEmbedder for ContrastiveWaveformEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> (Vec<f32>, usize) {
        if batch.num_spikes == 0 {
            return (Vec::new(), self.latent_dim);
        }
        let x = snippet_batch_to_tensor(batch, self.device).mul_scalar(0.01);
        let z = self.encode_normalized(&x);
        (z.into_vec(), self.latent_dim)
    }
}
