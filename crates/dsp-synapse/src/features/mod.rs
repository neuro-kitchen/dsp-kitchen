use dsp_base::linalg::PcaModel;
use crate::core::{FeatureEmbedder, SnippetBatch, WaveformSnippet};

pub mod morphology;

pub use morphology::{SpikeMorphology, compute_morphology};

/// Fits a PCA model (`dsp_base::linalg::PcaModel`) on a batch of spike waveforms and projects each spike into PCA feature coordinates.
pub fn extract_waveform_pca(
    snippets: &[WaveformSnippet],
    num_components: usize,
) -> Option<(PcaModel, Vec<Vec<f32>>)> {
    if snippets.is_empty() {
        return None;
    }

    let num_samples = snippets[0].num_samples;
    let num_spikes = snippets.len();

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

/// Classical PCA feature embedder delegating to [`dsp_base::linalg::PcaModel`].
#[derive(Debug, Clone, Copy)]
pub struct PcaFeatureEmbedder {
    pub num_components: usize,
}

impl Default for PcaFeatureEmbedder {
    fn default() -> Self {
        Self { num_components: 4 }
    }
}

impl FeatureEmbedder for PcaFeatureEmbedder {
    fn embed(&self, batch: &SnippetBatch) -> dsp_core::DspResult<(Vec<f32>, usize)> {
        let d = self.num_components.max(1);
        if batch.num_spikes == 0 {
            return Ok((Vec::new(), d));
        }

        let feat_len = batch.num_channels * batch.num_samples;
        let num_spikes = batch.num_spikes;

        let mut matrix = vec![0.0f32; feat_len * num_spikes];
        for s_idx in 0..num_spikes {
            let snip = batch.snippet_slice(s_idx);
            for (f, &val) in snip.iter().enumerate() {
                matrix[f * num_spikes + s_idx] = val;
            }
        }

        let pca = PcaModel::fit(&matrix, feat_len, num_spikes, d);
        let projected_flat = pca.project_cpu(&matrix, feat_len, num_spikes);
        let actual_d = pca.num_components;

        let mut out = vec![0.0f32; num_spikes * actual_d];
        for s_idx in 0..num_spikes {
            for k in 0..actual_d {
                out[s_idx * actual_d + k] = projected_flat[k * num_spikes + s_idx];
            }
        }
        Ok((out, actual_d))
    }
}
