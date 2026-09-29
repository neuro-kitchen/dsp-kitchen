use crate::extraction::WaveformSnippet;

/// Mean action potential template and standard deviation across a cluster of waveforms.
#[derive(Debug, Clone, PartialEq)]
pub struct WaveformTemplate {
    pub num_channels: usize,
    pub num_samples: usize,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

/// Computes the mean waveform template and standard deviation across a set of aligned waveforms.
pub fn compute_mean_template(snippets: &[WaveformSnippet]) -> Option<WaveformTemplate> {
    if snippets.is_empty() {
        return None;
    }

    let n_ch = snippets[0].num_channels();
    let n_s = snippets[0].num_samples;
    let total_elements = n_ch * n_s;
    let n_spikes = snippets.len() as f32;

    let mut mean = vec![0.0f32; total_elements];
    for snip in snippets {
        if snip.waveform.len() != total_elements {
            continue;
        }
        for i in 0..total_elements {
            mean[i] += snip.waveform[i];
        }
    }
    for val in mean.iter_mut() {
        *val /= n_spikes;
    }

    let mut std = vec![0.0f32; total_elements];
    for snip in snippets {
        if snip.waveform.len() != total_elements {
            continue;
        }
        for i in 0..total_elements {
            let diff = snip.waveform[i] - mean[i];
            std[i] += diff * diff;
        }
    }
    for val in std.iter_mut() {
        *val = (*val / n_spikes).sqrt();
    }

    Some(WaveformTemplate {
        num_channels: n_ch,
        num_samples: n_s,
        mean,
        std,
    })
}
