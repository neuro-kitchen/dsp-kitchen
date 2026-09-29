//! `EnsorArtifactRejector`: Neural stimulation and synchronous artifact rejection filter.
//!
//! Identifies and removes electrical stimulation pulses and common-mode photoelectric
//! transients across multi-channel `SnippetBatch` windows.

use anyhow::Result;
use dsp_synapse::SnippetBatch;
use crate::backbones::{LinearLayer, ResBlock1D};
use crate::backend::{SynapseMlDevice, snippet_batch_to_tensor};
use crate::hub::SafetensorsMap;

#[derive(Debug, Clone)]
pub struct EnsorArtifactRejector {
    pub res_block: ResBlock1D,
    pub head: LinearLayer,
    pub artifact_threshold: f32,
    pub device: SynapseMlDevice,
}

impl EnsorArtifactRejector {
    pub fn new(in_channels: usize, seed: u64, device: SynapseMlDevice) -> Self {
        Self {
            res_block: ResBlock1D::new(in_channels, 16, 5, seed, device),
            head: LinearLayer::new_initialized(16, 1, seed.wrapping_add(111), device),
            artifact_threshold: 0.70,
            device,
        }
    }

    /// Predicts artifact probability in `[0.0, 1.0]` for each snippet in `batch`.
    pub fn predict_artifact_scores(&self, batch: &SnippetBatch) -> Vec<f32> {
        if batch.num_spikes == 0 {
            return Vec::new();
        }
        let x = snippet_batch_to_tensor(batch, self.device).mul_scalar(0.01);
        let pooled = self.res_block.forward(&x).global_max_pool_1d();
        let probs = self.head.forward(&pooled).sigmoid();
        probs.data
    }

    /// Filters a `SnippetBatch`, retaining only spikes whose artifact score is below `artifact_threshold`.
    pub fn filter_clean_snippets(&self, batch: &SnippetBatch) -> SnippetBatch {
        if batch.num_spikes == 0 {
            return batch.clone();
        }
        let scores = self.predict_artifact_scores(batch);
        let k = batch.num_channels;
        let t = batch.num_samples;
        let stride = k * t;

        let mut data = Vec::new();
        let mut primary_channels = Vec::new();
        let mut center_samples = Vec::new();
        let mut subsample_offsets = Vec::new();
        let mut channel_ids = Vec::new();

        for (i, &score) in scores.iter().enumerate() {
            if score < self.artifact_threshold {
                data.extend_from_slice(&batch.data[i * stride..(i + 1) * stride]);
                primary_channels.push(batch.primary_channels[i]);
                center_samples.push(batch.center_samples[i]);
                subsample_offsets.push(batch.subsample_offsets[i]);
                channel_ids.extend_from_slice(batch.spike_channel_ids(i));
            }
        }

        let kept = primary_channels.len();
        SnippetBatch::from_raw_parts(
            data,
            kept,
            k,
            t,
            primary_channels,
            center_samples,
            subsample_offsets,
            channel_ids,
        )
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.res_block.save_weights("ensor.res_block", map);
        self.head.save_weights("ensor.head", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.res_block
            .load_weights("ensor.res_block", map, self.device)?;
        self.head.load_weights("ensor.head", map, self.device)?;
        Ok(())
    }
}
