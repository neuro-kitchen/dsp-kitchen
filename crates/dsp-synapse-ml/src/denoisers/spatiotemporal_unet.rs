//! `SpatiotemporalUnetDenoiser`: Multi-channel $K$-NN U-Net waveform denoiser with
//! cross-electrode attention.
//!
//! Implements `dsp_synapse::traits::WaveformDenoiser`.

use anyhow::Result;
use dsp_synapse::{SnippetBatch, WaveformDenoiser};
use crate::backbones::{CrossElectrodeAttention, UNet1DBackbone};
use crate::backend::{
    SynapseMlDevice, Tensor3D, snippet_batch_to_tensor, tensor_to_snippet_batch,
};
use crate::hub::SafetensorsMap;

#[derive(Debug, Clone)]
pub struct SpatiotemporalUnetDenoiser {
    pub attention: CrossElectrodeAttention,
    pub unet: UNet1DBackbone,
    pub residual_blend: f32,
    pub device: SynapseMlDevice,
}

impl SpatiotemporalUnetDenoiser {
    pub fn new(
        k_channels: usize,
        snippet_samples: usize,
        base_channels: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        Self {
            attention: CrossElectrodeAttention::new(snippet_samples, seed, device),
            unet: UNet1DBackbone::new(
                k_channels,
                k_channels,
                base_channels,
                seed.wrapping_add(1_000),
                device,
            ),
            residual_blend: 0.30,
            device,
        }
    }

    pub fn forward(&self, input: &Tensor3D) -> Tensor3D {
        let attn_feat = self.attention.forward(input);
        let noise_pred = self
            .unet
            .forward(&attn_feat)
            .mul_scalar(self.residual_blend);
        input.sub(&noise_pred)
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.attention.save_weights("st_unet.attention", map);
        self.unet.save_weights("st_unet.unet", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.attention
            .load_weights("st_unet.attention", map, self.device)?;
        self.unet.load_weights("st_unet.unet", map, self.device)?;
        Ok(())
    }
}

impl WaveformDenoiser for SpatiotemporalUnetDenoiser {
    fn denoise(&self, batch: &SnippetBatch) -> dsp_core::DspResult<SnippetBatch> {
        if batch.num_spikes == 0 {
            return Ok(batch.clone());
        }
        let x = snippet_batch_to_tensor(batch, self.device);
        let y = self.forward(&x);
        Ok(tensor_to_snippet_batch(y, batch))
    }
}
