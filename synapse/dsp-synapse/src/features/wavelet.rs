//! Multi-level Discrete Wavelet Transform (DWT) Feature Extraction (`wavelet.rs`, *Wave_Clus* style).
//!
//! Decomposes each spike snippet with a multi-level Haar filter bank and selects the $D$ wavelet
//! coefficients with the largest spread across the spike population, ranked by a normalized
//! interquartile range (Wave_Clus uses a Lilliefors test; this is a simpler proxy).

use dsp_base::math::interquartile_range;
use crate::core::{FeatureEmbedder, SnippetBatch};

/// Computes a multi-level 1D Haar Discrete Wavelet Transform (DWT) of `signal`.
pub fn haar_dwt_multilevel_1d(signal: &[f32], levels: usize) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }
    let mut coeffs = signal.to_vec();
    let mut len = coeffs.len();
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    for _ in 0..levels.max(1) {
        let half = len / 2;
        if half == 0 {
            break;
        }
        let mut temp = vec![0.0f32; half * 2];
        for i in 0..half {
            let a = coeffs[2 * i];
            let b = coeffs[2 * i + 1];
            temp[i] = (a + b) * inv_sqrt2;
            temp[half + i] = (a - b) * inv_sqrt2;
        }
        coeffs[..half * 2].copy_from_slice(&temp);
        len = half;
    }
    coeffs
}

/// Wavelet feature embedder (*Wave_Clus* / Quiroga et al., 2004) implementing [`FeatureEmbedder`].
#[derive(Debug, Clone, Copy)]
pub struct WaveletFeatureEmbedder {
    pub num_components: usize,
    pub levels: usize,
}

impl Default for WaveletFeatureEmbedder {
    fn default() -> Self {
        Self {
            num_components: 8,
            levels: 4,
        }
    }
}

impl FeatureEmbedder for WaveletFeatureEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> dsp_core::DspResult<(Vec<f32>, usize)> {
        let d = self.num_components.max(1);
        if batch.num_spikes == 0 {
            return Ok((Vec::new(), d));
        }

        let num_spikes = batch.num_spikes;
        let ch_count = batch.num_channels;
        let t_len = batch.num_samples;
        let total_coeffs = ch_count * t_len;
        let actual_d = d.min(total_coeffs).max(1);

        // 1. Compute DWT coefficients for every spike across all snippet channels
        let mut all_coeffs = vec![0.0f32; num_spikes * total_coeffs];
        for s in 0..num_spikes {
            for c in 0..ch_count {
                let wave = batch.channel_slice(s, c);
                let dwt = haar_dwt_multilevel_1d(wave, self.levels);
                let dst = &mut all_coeffs[s * total_coeffs + c * t_len..s * total_coeffs + (c + 1) * t_len];
                dst.copy_from_slice(&dwt);
            }
        }

        // 2. Score each coefficient index by robust IQR across spikes
        let mut col_buf = vec![0.0f32; num_spikes];
        let mut scores: Vec<(usize, f32)> = (0..total_coeffs)
            .map(|f| {
                for s in 0..num_spikes {
                    col_buf[s] = all_coeffs[s * total_coeffs + f];
                }
                (f, interquartile_range(&col_buf))
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: Vec<usize> = scores.iter().take(actual_d).map(|&(idx, _)| idx).collect();
        let mut out = vec![0.0f32; num_spikes * actual_d];
        for s in 0..num_spikes {
            for (k, &f_idx) in selected.iter().enumerate() {
                out[s * actual_d + k] = all_coeffs[s * total_coeffs + f_idx];
            }
        }

        Ok((out, actual_d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_dwt_preserves_energy() {
        let sig = vec![1.0f32, -2.0, 4.0, -1.0, 3.0, 0.5, -1.5, 2.0];
        let e_in: f32 = sig.iter().map(|v| v * v).sum();
        let coeffs = haar_dwt_multilevel_1d(&sig, 3);
        let e_out: f32 = coeffs.iter().map(|v| v * v).sum();
        assert!((e_in - e_out).abs() < 1e-4);
    }
}
