//! `YassNeuralDetector`: Spatiotemporal convolutional neural spike detector inspired by YASS
//! (Lee et al., *Yet Another Spike Sorter*).
//!
//! Applies temporal matched-feature convolutions followed by pointwise probability scoring
//! and refractory local-maximum selection.

use anyhow::Result;
use dsp_synapse::{SpikeDetector, SpikeEvent};
use crate::backbones::Conv1dLayer;
use crate::backend::{SynapseMlDevice, Tensor3D};
use crate::hub::SafetensorsMap;

#[derive(Debug, Clone)]
pub struct YassNeuralDetector {
    pub conv_temporal: Conv1dLayer,
    pub conv_hidden: Conv1dLayer,
    pub conv_score: Conv1dLayer,
    pub probability_threshold: f32,
    pub device: SynapseMlDevice,
}

impl YassNeuralDetector {
    pub fn new(hidden_channels: usize, seed: u64, device: SynapseMlDevice) -> Self {
        let h = hidden_channels.max(8);
        let mut conv_temporal = Conv1dLayer::new_initialized(1, h, 7, 1, 3, seed, device);
        // Set filter 0 to respond strongly to negative troughs
        conv_temporal.weight.data[3] = -2.0;

        let conv_hidden =
            Conv1dLayer::new_initialized(h, h, 3, 1, 1, seed.wrapping_add(100), device);
        let mut conv_score =
            Conv1dLayer::new_initialized(h, 1, 1, 1, 0, seed.wrapping_add(200), device);
        conv_score.bias.data[0] = -1.0;

        Self {
            conv_temporal,
            conv_hidden,
            conv_score,
            probability_threshold: 0.5,
            device,
        }
    }

    /// Computes per-sample spike probability map `[N_channels, 1, Samples]`.
    pub fn score_channels(&self, channel_signals: &Tensor3D) -> Tensor3D {
        let h1 = self.conv_temporal.forward(channel_signals).relu();
        let h2 = self.conv_hidden.forward(&h1).relu();
        self.conv_score.forward(&h2).sigmoid()
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.conv_temporal.save_weights("yass.conv_temporal", map);
        self.conv_hidden.save_weights("yass.conv_hidden", map);
        self.conv_score.save_weights("yass.conv_score", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.conv_temporal
            .load_weights("yass.conv_temporal", map, self.device)?;
        self.conv_hidden
            .load_weights("yass.conv_hidden", map, self.device)?;
        self.conv_score
            .load_weights("yass.conv_score", map, self.device)?;
        Ok(())
    }
}

impl SpikeDetector for YassNeuralDetector {
    fn detect(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        sample_rate_hz: f64,
    ) -> Vec<SpikeEvent> {
        if channels == 0 || samples < 8 {
            return Vec::new();
        }
        assert_eq!(data.len(), channels * samples);

        // Normalize input voltage by 50 uV scale into [channels, 1, samples]
        let scaled: Vec<f32> = data.iter().map(|&v| v / 50.0).collect();
        let input = Tensor3D::from_floats(scaled, [channels, 1, samples], self.device);
        let prob_map = self.score_channels(&input);

        let refractory_samples = ((sample_rate_hz * 0.001) as usize).max(10);
        let mut events = Vec::new();

        for ch in 0..channels {
            let off = ch * samples;
            let p_slice = &prob_map.data[off..off + samples];
            let raw_slice = &data[off..off + samples];
            let mut last_t = 0usize;

            for t in 1..samples.saturating_sub(1) {
                let p = p_slice[t];
                if p >= self.probability_threshold
                    && raw_slice[t] < 0.0
                    && raw_slice[t] < raw_slice[t - 1]
                    && raw_slice[t] <= raw_slice[t + 1]
                {
                    if events.is_empty() || t > last_t + refractory_samples {
                        events.push(SpikeEvent {
                            channel_id: ch,
                            sample_index: t as u64,
                            peak_amplitude_uv: raw_slice[t],
                        });
                        last_t = t;
                    }
                }
            }
        }

        events.sort_by_key(|e| e.sample_index);
        events
    }
}
