use serde::{Deserialize, Serialize};
use super::snippets::WaveformSnippet;

/// Automated single-unit curation classification labels (Allen / IBL standard).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitQualityLabel {
    /// Well-isolated single biological neuron (SUA)
    SingleUnit,
    /// Multi-unit activity or overlapping cluster (MUA)
    MultiUnit,
    /// Non-biological electrical/motion artifact or thermal noise
    Noise,
}

/// Mean action potential template, standard deviation (`std`), and standard error of the mean (`se`)
/// across a cluster of waveforms.
///
/// Row `r` of `mean` / `std` / `se` (each `num_samples` long) belongs to recording channel
/// `channel_ids[r]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveformTemplate {
    /// Recording channel of each row.
    pub channel_ids: Vec<usize>,
    pub num_channels: usize,
    pub num_samples: usize,
    /// Number of waveforms accumulated into this template (`>= 1`).
    pub count: usize,
    /// Sample of the deepest trough (minimum of `mean` over all rows).
    pub trough_index: usize,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    /// Standard Error of the Mean (`SE = SD / sqrt(count)`).
    pub se: Vec<f32>,
}

impl WaveformTemplate {
    /// Builds a template; `mean` and `std` are `[channel_ids.len(), num_samples]`.
    pub fn new(channel_ids: Vec<usize>, num_samples: usize, mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self::with_count(channel_ids, num_samples, 1, mean, std)
    }

    /// Builds a template with explicit waveform `count`, automatically computing `se = std / sqrt(count)`.
    pub fn with_count(
        channel_ids: Vec<usize>,
        num_samples: usize,
        count: usize,
        mean: Vec<f32>,
        std: Vec<f32>,
    ) -> Self {
        let num_channels = channel_ids.len();
        assert_eq!(mean.len(), num_channels * num_samples, "mean shape");
        assert_eq!(std.len(), num_channels * num_samples, "std shape");
        let trough_index = mean
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i % num_samples.max(1));
        let denom = (count.max(1) as f32).sqrt();
        let se = std.iter().map(|&s| s / denom).collect();
        Self {
            channel_ids,
            num_channels,
            num_samples,
            count: count.max(1),
            trough_index,
            mean,
            std,
            se,
        }
    }

    /// Mean waveform of row `r`.
    pub fn row(&self, r: usize) -> &[f32] {
        &self.mean[r * self.num_samples..(r + 1) * self.num_samples]
    }

    /// Standard error waveform of row `r`.
    pub fn se_row(&self, r: usize) -> &[f32] {
        &self.se[r * self.num_samples..(r + 1) * self.num_samples]
    }

    /// Mean waveform on recording channel `channel`, if the template covers it.
    pub fn channel_row(&self, channel: usize) -> Option<&[f32]> {
        self.channel_ids.iter().position(|&c| c == channel).map(|r| self.row(r))
    }
}

/// Computes the mean waveform template, standard deviation, and standard error (`SE = SD / sqrt(n)`)
/// across a set of aligned waveforms.
/// Snippets on other channels (or with another shape) than the first one are skipped.
pub fn compute_mean_template(snippets: &[WaveformSnippet]) -> Option<WaveformTemplate> {
    let first = snippets.first()?;
    let n_s = first.num_samples;
    let total_elements = first.num_channels() * n_s;
    let used: Vec<&WaveformSnippet> = snippets
        .iter()
        .filter(|s| s.waveform.len() == total_elements && s.channel_ids == first.channel_ids)
        .collect();
    if used.is_empty() {
        return None;
    }
    let count = used.len();
    let n = count as f32;

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

    Some(WaveformTemplate::with_count(
        first.channel_ids.clone(),
        n_s,
        count,
        mean,
        std,
    ))
}
