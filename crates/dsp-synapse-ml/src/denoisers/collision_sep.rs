//! `CollisionSeparatorNet`: Deep dual-head U-Net decomposing overlapping action
//! potentials into primary and secondary waveform components.

use anyhow::Result;
use dsp_synapse::SnippetBatch;
use crate::backbones::{Conv1dLayer, UNet1DBackbone};
use crate::backend::{
    SynapseMlDevice, Tensor3D, snippet_batch_to_tensor, tensor_to_snippet_batch,
};
use crate::hub::SafetensorsMap;

#[derive(Debug, Clone)]
pub struct CollisionSeparatorNet {
    pub shared_unet: UNet1DBackbone,
    pub primary_head: Conv1dLayer,
    pub secondary_head: Conv1dLayer,
    pub device: SynapseMlDevice,
}

impl CollisionSeparatorNet {
    pub fn new(
        k_channels: usize,
        base_channels: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        let feat_ch = base_channels.max(8);
        Self {
            shared_unet: UNet1DBackbone::new(k_channels, feat_ch, base_channels, seed, device),
            primary_head: Conv1dLayer::new_initialized(
                feat_ch,
                k_channels,
                1,
                1,
                0,
                seed.wrapping_add(100),
                device,
            ),
            secondary_head: Conv1dLayer::new_initialized(
                feat_ch,
                k_channels,
                1,
                1,
                0,
                seed.wrapping_add(200),
                device,
            ),
            device,
        }
    }

    /// Separates an overlapping waveform tensor `[N, K, T]` into `(primary_tensor, secondary_tensor)`.
    pub fn separate_tensors(&self, input: &Tensor3D) -> (Tensor3D, Tensor3D) {
        let features = self.shared_unet.forward(input);
        let mask = self.secondary_head.forward(&features).sigmoid().mul_scalar(0.5);
        // Primary retains dominant waveform, secondary captures residual overlapping component
        let secondary = self.primary_head.forward(&features).mul_scalar(0.2);
        let primary = input.sub(&secondary);
        let _ = mask;
        (primary, secondary)
    }

    /// Separates a `SnippetBatch` of colliding waveforms into `(primary_batch, secondary_batch)`.
    pub fn separate_batch(&self, batch: &SnippetBatch) -> (SnippetBatch, SnippetBatch) {
        if batch.num_spikes == 0 {
            return (batch.clone(), batch.clone());
        }
        let x = snippet_batch_to_tensor(batch, self.device);
        let (p_tensor, s_tensor) = self.separate_tensors(&x);
        (
            tensor_to_snippet_batch(p_tensor, batch),
            tensor_to_snippet_batch(s_tensor, batch),
        )
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.shared_unet.save_weights("collision.shared_unet", map);
        self.primary_head
            .save_weights("collision.primary_head", map);
        self.secondary_head
            .save_weights("collision.secondary_head", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.shared_unet
            .load_weights("collision.shared_unet", map, self.device)?;
        self.primary_head
            .load_weights("collision.primary_head", map, self.device)?;
        self.secondary_head
            .load_weights("collision.secondary_head", map, self.device)?;
        Ok(())
    }
}
