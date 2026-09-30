//! `SpikeDeeptector`: 1D-CNN classifier separating biological action potentials
//! from non-biological artifacts and background thermal noise.
//!
//! Implements `dsp_synapse::traits::SpikeDetector`.

use anyhow::Result;
use dsp_synapse::{SpikeDetector, SpikeEvent, SnippetBatch, detect_spikes_multichannel};
use crate::backbones::{LinearLayer, ResBlock1D};
use crate::backend::{SynapseMlDevice, Tensor2D, Tensor3D, snippet_batch_to_tensor};
use crate::hub::SafetensorsMap;

/// Class probabilities output by `SpikeDeeptector` for a candidate waveform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeClassProbabilities {
    pub p_noise: f32,
    pub p_artifact: f32,
    pub p_spike: f32,
}

/// 1D-CNN Spike vs. Artifact detector & classifier.
#[derive(Debug, Clone)]
pub struct SpikeDeeptector {
    pub block1: ResBlock1D,
    pub block2: ResBlock1D,
    pub classifier: LinearLayer,
    pub snippet_samples: usize,
    pub pre_samples: usize,
    pub spike_prob_threshold: f32,
    pub candidate_threshold_sigma: f32,
    pub device: SynapseMlDevice,
}

impl SpikeDeeptector {
    pub fn new(
        in_channels: usize,
        snippet_samples: usize,
        seed: u64,
        device: SynapseMlDevice,
    ) -> Self {
        let mut model = Self {
            block1: ResBlock1D::new(in_channels, 16, 5, seed, device),
            block2: ResBlock1D::new(16, 32, 3, seed.wrapping_add(500), device),
            classifier: LinearLayer::new_initialized(32, 3, seed.wrapping_add(900), device),
            snippet_samples: snippet_samples.max(16),
            pre_samples: (snippet_samples / 3).max(5),
            spike_prob_threshold: 0.45,
            candidate_threshold_sigma: 4.0,
            device,
        };
        // Bias class 2 (biological spike) slightly positively so negative-trough waves activate strongly
        model.classifier.bias.data[2] = 0.5;
        model
    }

    /// Predicts `[P(noise), P(artifact), P(spike)]` for a 3D batch `[N, C, T]`.
    pub fn predict_proba_tensor(&self, input: &Tensor3D) -> Tensor2D {
        let h1 = self.block1.forward(input).max_pool1d(2, 2);
        let h2 = self.block2.forward(&h1).global_avg_pool_1d();
        self.classifier.forward(&h2).softmax()
    }

    /// Predicts class probabilities for each snippet in a `SnippetBatch`.
    pub fn classify_batch(&self, batch: &SnippetBatch) -> Vec<SpikeClassProbabilities> {
        if batch.num_spikes == 0 {
            return Vec::new();
        }
        let tensor = snippet_batch_to_tensor(batch, self.device);
        let probs = self.predict_proba_tensor(&tensor);
        let mut out = Vec::with_capacity(batch.num_spikes);
        for i in 0..batch.num_spikes {
            out.push(SpikeClassProbabilities {
                p_noise: probs.data[i * 3],
                p_artifact: probs.data[i * 3 + 1],
                p_spike: probs.data[i * 3 + 2],
            });
        }
        out
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.block1.save_weights("spikedeeptector.block1", map);
        self.block2.save_weights("spikedeeptector.block2", map);
        self.classifier
            .save_weights("spikedeeptector.classifier", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.block1
            .load_weights("spikedeeptector.block1", map, self.device)?;
        self.block2
            .load_weights("spikedeeptector.block2", map, self.device)?;
        self.classifier
            .load_weights("spikedeeptector.classifier", map, self.device)?;
        Ok(())
    }
}

impl SpikeDetector for SpikeDeeptector {
    fn detect(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        sample_rate_hz: f64,
    ) -> dsp_core::DspResult<Vec<SpikeEvent>> {
        let refractory_samples = ((sample_rate_hz * 0.001) as usize).max(10);
        // 1. Propose candidates via fast threshold crossing
        let candidates = detect_spikes_multichannel(
            data,
            channels,
            samples,
            self.candidate_threshold_sigma,
            refractory_samples,
        );
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let post_samples = self.snippet_samples.saturating_sub(self.pre_samples);
        let mut valid_candidates = Vec::with_capacity(candidates.len());
        let mut flat_snippets = Vec::with_capacity(candidates.len() * self.snippet_samples);

        for ev in candidates {
            let center = ev.sample_index as usize;
            if center < self.pre_samples || center + post_samples > samples {
                continue;
            }
            let ch_off = ev.channel_id * samples;
            let start = ch_off + center - self.pre_samples;
            // Normalize snippet by 100 uV scale for stable neural activations
            for &v in &data[start..start + self.snippet_samples] {
                flat_snippets.push(v / 100.0);
            }
            valid_candidates.push(ev);
        }

        if valid_candidates.is_empty() {
            return Ok(Vec::new());
        }

        let n = valid_candidates.len();
        let input_tensor =
            Tensor3D::from_floats(flat_snippets, [n, 1, self.snippet_samples], self.device);
        let probs = self.predict_proba_tensor(&input_tensor);

        Ok(valid_candidates
            .into_iter()
            .enumerate()
            .filter_map(|(i, ev)| {
                let p_spike = probs.data[i * 3 + 2];
                if p_spike >= self.spike_prob_threshold {
                    Some(ev)
                } else {
                    None
                }
            })
            .collect())
    }
}
