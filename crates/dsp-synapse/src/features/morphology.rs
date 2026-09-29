use serde::{Deserialize, Serialize};
use crate::extraction::WaveformSnippet;

/// Morphological feature metrics extracted from an action potential waveform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeMorphology {
    pub peak_amplitude_uv: f32,
    pub trough_to_peak_samples: f32,
    pub half_width_samples: f32,
    pub repolarization_slope: f32,
}

/// Computes morphological waveform features from a single-channel snippet.
pub fn compute_morphology(snippet: &WaveformSnippet) -> Option<SpikeMorphology> {
    if snippet.waveform.is_empty() {
        return None;
    }

    let w = &snippet.waveform;
    // 1. Trough (minimum value)
    let (trough_idx, &trough_val) = w
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;

    // 2. Subsequent peak (maximum value after trough)
    if trough_idx >= w.len() - 1 {
        return None;
    }

    let (peak_offset, &peak_val) = w[trough_idx..]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;

    let peak_idx = trough_idx + peak_offset;
    let trough_to_peak = peak_idx as f32 - trough_idx as f32;

    // 3. Half-width at half-minimum
    let half_amp = trough_val * 0.5;
    let mut left_idx = trough_idx as f32;
    for i in (0..trough_idx).rev() {
        if w[i] > half_amp {
            left_idx = i as f32;
            break;
        }
    }
    let mut right_idx = trough_idx as f32;
    for i in trough_idx..w.len() {
        if w[i] > half_amp {
            right_idx = i as f32;
            break;
        }
    }
    let half_width = (right_idx - left_idx).max(1.0);

    // 4. Repolarization slope (dV/dt between trough and peak)
    let repol_slope = if trough_to_peak > 0.0 {
        (peak_val - trough_val) / trough_to_peak
    } else {
        0.0
    };

    Some(SpikeMorphology {
        peak_amplitude_uv: trough_val,
        trough_to_peak_samples: trough_to_peak,
        half_width_samples: half_width,
        repolarization_slope: repol_slope,
    })
}
