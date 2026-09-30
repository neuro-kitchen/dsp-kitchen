use crate::extraction::WaveformSnippet;

/// Mean action potential template and standard deviation across a cluster of waveforms.
///
/// Row `r` of `mean` / `std` (each `num_samples` long) belongs to recording channel
/// `channel_ids[r]`.
#[derive(Debug, Clone, PartialEq)]
pub struct WaveformTemplate {
    /// Recording channel of each row.
    pub channel_ids: Vec<usize>,
    pub num_channels: usize,
    pub num_samples: usize,
    /// Sample of the deepest trough (minimum of `mean` over all rows).
    pub trough_index: usize,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl WaveformTemplate {
    /// Builds a template; `mean` and `std` are `[channel_ids.len(), num_samples]`.
    pub fn new(channel_ids: Vec<usize>, num_samples: usize, mean: Vec<f32>, std: Vec<f32>) -> Self {
        let num_channels = channel_ids.len();
        assert_eq!(mean.len(), num_channels * num_samples, "mean shape");
        assert_eq!(std.len(), num_channels * num_samples, "std shape");
        let trough_index = mean
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i % num_samples.max(1));
        Self { channel_ids, num_channels, num_samples, trough_index, mean, std }
    }

    /// Mean waveform of row `r`.
    pub fn row(&self, r: usize) -> &[f32] {
        &self.mean[r * self.num_samples..(r + 1) * self.num_samples]
    }

    /// Mean waveform on recording channel `channel`, if the template covers it.
    pub fn channel_row(&self, channel: usize) -> Option<&[f32]> {
        self.channel_ids.iter().position(|&c| c == channel).map(|r| self.row(r))
    }
}

/// Computes the mean waveform template and standard deviation across a set of aligned waveforms.
/// Snippets on other channels (or with another shape) than the first one are skipped.
pub fn compute_mean_template(snippets: &[WaveformSnippet]) -> Option<WaveformTemplate> {
    let first = snippets.first()?;
    let n_s = first.num_samples;
    let total_elements = first.num_channels() * n_s;
    let used: Vec<&WaveformSnippet> = snippets
        .iter()
        .filter(|s| s.waveform.len() == total_elements && s.channel_ids == first.channel_ids)
        .collect();
    let n = used.len() as f32;

    let mut mean = vec![0.0f32; total_elements];
    for snip in &used {
        for (m, v) in mean.iter_mut().zip(&snip.waveform) {
            *m += v;
        }
    }
    mean.iter_mut().for_each(|m| *m /= n);

    let mut std = vec![0.0f32; total_elements];
    for snip in &used {
        for ((s, v), m) in std.iter_mut().zip(&snip.waveform).zip(&mean) {
            *s += (v - m) * (v - m);
        }
    }
    std.iter_mut().for_each(|s| *s = (*s / n).sqrt());

    Some(WaveformTemplate::new(first.channel_ids.clone(), n_s, mean, std))
}
