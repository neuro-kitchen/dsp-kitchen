use serde::{Deserialize, Serialize};
use dsp_base::linalg::PcaModel;
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

/// Fits a PCA model on a batch of spike waveforms and projects each spike into PCA feature coordinates.
pub fn extract_waveform_pca(
    snippets: &[WaveformSnippet],
    num_components: usize,
) -> Option<(PcaModel, Vec<Vec<f32>>)> {
    if snippets.is_empty() {
        return None;
    }

    let num_samples = snippets[0].num_samples;
    let num_spikes = snippets.len();

    // Flatten into matrix: [num_samples, num_spikes] (each spike is a sample vector)
    let mut matrix = vec![0.0f32; num_samples * num_spikes];
    for (s_idx, snip) in snippets.iter().enumerate() {
        if snip.waveform.len() != num_samples {
            continue;
        }
        for t in 0..num_samples {
            matrix[t * num_spikes + s_idx] = snip.waveform[t];
        }
    }

    let pca = PcaModel::fit(&matrix, num_samples, num_spikes, num_components);
    let projected_flat = pca.project_cpu(&matrix, num_samples, num_spikes);

    // Group projected coordinates per spike: [num_spikes][num_components]
    let mut per_spike_pcs = Vec::with_capacity(num_spikes);
    for s_idx in 0..num_spikes {
        let mut pcs = Vec::with_capacity(num_components);
        for k in 0..pca.num_components {
            pcs.push(projected_flat[k * num_spikes + s_idx]);
        }
        per_spike_pcs.push(pcs);
    }

    Some((pca, per_spike_pcs))
}
