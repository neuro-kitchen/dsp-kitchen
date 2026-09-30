//! `SingleChannelDenoiser`: SpikeInterface / YASS 1D deep residual waveform denoiser.
//!
//! Operates independently on each electrode channel `[N * K, 1, T]` to suppress
//! high-frequency thermal noise while preserving peak trough morphology.
//! Implements `dsp_synapse::traits::WaveformDenoiser`.

use anyhow::Result;
use dsp_synapse::{SnippetBatch, WaveformDenoiser};
use crate::backbones::{Conv1dLayer, ResBlock1D};
use crate::backend::{SynapseMlDevice, Tensor3D};
use crate::hub::SafetensorsMap;

#[derive(Debug, Clone)]
pub struct SingleChannelDenoiser {
    pub stem: Conv1dLayer,
    pub res_block: ResBlock1D,
    pub noise_head: Conv1dLayer,
    pub residual_blend: f32,
    pub device: SynapseMlDevice,
}

impl SingleChannelDenoiser {
    pub fn new(hidden_channels: usize, seed: u64, device: SynapseMlDevice) -> Self {
        let h = hidden_channels.max(8);
        Self {
            stem: Conv1dLayer::new_initialized(1, h, 5, 1, 2, seed, device),
            res_block: ResBlock1D::new(h, h, 3, seed.wrapping_add(100), device),
            noise_head: Conv1dLayer::new_initialized(h, 1, 3, 1, 1, seed.wrapping_add(200), device),
            residual_blend: 0.25,
            device,
        }
    }

    /// Denoises a 3D tensor `[B, 1, T]` via residual noise prediction (`x - alpha * f(x)`).
    pub fn forward_single_channel(&self, input: &Tensor3D) -> Tensor3D {
        let h = self.stem.forward(input).gelu();
        let r = self.res_block.forward(&h);
        let noise_est = self
            .noise_head
            .forward(&r)
            .match_temporal_length(input.shape[2])
            .mul_scalar(self.residual_blend);
        input.sub(&noise_est)
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.stem.save_weights("sc_denoiser.stem", map);
        self.res_block.save_weights("sc_denoiser.res_block", map);
        self.noise_head.save_weights("sc_denoiser.noise_head", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.stem
            .load_weights("sc_denoiser.stem", map, self.device)?;
        self.res_block
            .load_weights("sc_denoiser.res_block", map, self.device)?;
        self.noise_head
            .load_weights("sc_denoiser.noise_head", map, self.device)?;
        Ok(())
    }
}

impl WaveformDenoiser for SingleChannelDenoiser {
    fn denoise(&self, batch: &SnippetBatch) -> dsp_core::DspResult<SnippetBatch> {
        if batch.num_spikes == 0 {
            return Ok(batch.clone());
        }
        let n = batch.num_spikes;
        let k = batch.num_channels;
        let t = batch.num_samples;

        // Reshape [N, K, T] -> [N * K, 1, T]
        let flat_in = Tensor3D::from_floats(batch.data.clone(), [n * k, 1, t], self.device);
        let denoised = self.forward_single_channel(&flat_in);

        Ok(SnippetBatch::from_raw_parts(
            denoised.into_vec(),
            n,
            k,
            t,
            batch.primary_channels.clone(),
            batch.center_samples.clone(),
            batch.subsample_offsets.clone(),
            batch.channel_ids.clone(),
        ))
    }
}
