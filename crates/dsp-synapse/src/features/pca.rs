use dsp_base::linalg::PcaModel;
use crate::extraction::WaveformSnippet;

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

    // Flatten into matrix: [num_samples, num_spikes]
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
