//! `UnitQualityClassifier`: Automated neural quality control model classifying sorted
//! units into `SingleUnit (SUA)`, `MultiUnit (MUA)`, or `Noise` based on Allen / IBL
//! quality metrics and template morphology.

use anyhow::Result;
use dsp_synapse::UnitQualityLabel;
use serde::{Deserialize, Serialize};
use crate::backbones::MlpBackbone;
use crate::backend::{SynapseMlDevice, Tensor2D};
use crate::hub::SafetensorsMap;

/// 8-feature vector summarizing a sorted unit's quality and waveform morphology.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UnitQualityFeatures {
    pub snr: f32,
    pub isi_violation_rate_pct: f32,
    pub firing_rate_hz: f32,
    pub amplitude_cutoff: f32,
    pub presence_ratio: f32,
    pub half_width_ms: f32,
    pub trough_to_peak_ms: f32,
    pub repolarization_slope: f32,
}

impl UnitQualityFeatures {
    pub fn to_normalized_array(&self) -> [f32; 8] {
        [
            self.snr / 10.0,
            self.isi_violation_rate_pct,
            (self.firing_rate_hz / 20.0).min(5.0),
            self.amplitude_cutoff * 10.0,
            self.presence_ratio,
            self.half_width_ms,
            self.trough_to_peak_ms,
            self.repolarization_slope / 100.0,
        ]
    }
}

/// Automated unit curation prediction with probabilities `[P(SUA), P(MUA), P(Noise)]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UnitCurationPrediction {
    pub label: UnitQualityLabel,
    pub p_single_unit: f32,
    pub p_multi_unit: f32,
    pub p_noise: f32,
}

#[derive(Debug, Clone)]
pub struct UnitQualityClassifier {
    pub mlp: MlpBackbone,
    pub device: SynapseMlDevice,
}

impl UnitQualityClassifier {
    pub fn new(seed: u64, device: SynapseMlDevice) -> Self {
        let mlp = MlpBackbone::new(8, &[16, 16], 3, seed, device);
        Self { mlp, device }
    }

    /// Classifies a slice of unit quality feature vectors into `SUA`, `MUA`, or `Noise`.
    pub fn classify_units(&self, units: &[UnitQualityFeatures]) -> Vec<UnitCurationPrediction> {
        let n = units.len();
        if n == 0 {
            return Vec::new();
        }

        let mut flat = Vec::with_capacity(n * 8);
        for u in units {
            flat.extend_from_slice(&u.to_normalized_array());
        }

        let x = Tensor2D::from_floats(flat, [n, 8], self.device);
        let mut logits = self.mlp.forward(&x);

        // Combine MLP learned logits with Allen Institute / IBL biophysical priors
        for (i, u) in units.iter().enumerate() {
            let sua_prior = (u.snr - 3.0) * 0.8
                - u.isi_violation_rate_pct * 3.0
                - u.amplitude_cutoff * 5.0
                + (u.presence_ratio - 0.8) * 2.0;
            let mua_prior = u.isi_violation_rate_pct * 2.5 + (u.snr - 2.0) * 0.3;
            let noise_prior = (2.5 - u.snr) * 1.5 + (0.3 - u.presence_ratio) * 2.0;

            logits.data[i * 3] += sua_prior;
            logits.data[i * 3 + 1] += mua_prior;
            logits.data[i * 3 + 2] += noise_prior;
        }

        let probs = logits.softmax();
        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            let p_sua = probs.data[i * 3];
            let p_mua = probs.data[i * 3 + 1];
            let p_noise = probs.data[i * 3 + 2];

            let label = if p_sua >= p_mua && p_sua >= p_noise {
                UnitQualityLabel::SingleUnit
            } else if p_mua >= p_noise {
                UnitQualityLabel::MultiUnit
            } else {
                UnitQualityLabel::Noise
            };

            results.push(UnitCurationPrediction {
                label,
                p_single_unit: p_sua,
                p_multi_unit: p_mua,
                p_noise,
            });
        }
        results
    }

    pub fn save_weights(&self, map: &mut SafetensorsMap) {
        self.mlp.save_weights("unit_classifier.mlp", map);
    }

    pub fn load_weights(&mut self, map: &SafetensorsMap) -> Result<()> {
        self.mlp
            .load_weights("unit_classifier.mlp", map, self.device)
    }
}
