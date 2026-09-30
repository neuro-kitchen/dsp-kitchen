//! Online Welford accumulator for multi-channel waveform templates in $O(1)$ memory.

use crate::extraction::WaveformSnippet;
use crate::metrics::WaveformTemplate;

/// Per-channel / per-unit online Welford accumulator computing exact running mean and
/// standard deviation across an arbitrary stream of [`WaveformSnippet`]s without storing
/// individual snippets in RAM.
#[derive(Debug, Clone)]
pub struct TemplateAccumulator {
    num_channels: usize,
    num_samples: usize,
    count: u64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl TemplateAccumulator {
    /// Creates a new accumulator for snippets of shape `[num_channels, num_samples]`.
    pub fn new(num_channels: usize, num_samples: usize) -> Self {
        let len = num_channels * num_samples;
        Self {
            num_channels,
            num_samples,
            count: 0,
            mean: vec![0.0f64; len],
            m2: vec![0.0f64; len],
        }
    }

    /// Number of waveforms accumulated so far.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Incorporates a single [`WaveformSnippet`] into the running mean and variance ($M_2$).
    pub fn update(&mut self, snippet: &WaveformSnippet) {
        let expected = self.num_channels * self.num_samples;
        if snippet.waveform.len() != expected {
            return;
        }

        self.count += 1;
        let n = self.count as f64;

        for (i, &val) in snippet.waveform.iter().enumerate() {
            let x = val as f64;
            let delta = x - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = x - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Merges a GPU-reduced batch of `batch_count` waveforms given their per-sample `batch_sum`
    /// ($\sum x$) and `batch_sum_sq` ($\sum x^2$) using Chan's parallel variance update formula.
    pub fn merge_batch(&mut self, batch_count: u64, batch_sum: &[f32], batch_sum_sq: &[f32]) {
        let expected = self.num_channels * self.num_samples;
        if batch_count == 0 || batch_sum.len() != expected || batch_sum_sq.len() != expected {
            return;
        }

        let n_a = self.count as f64;
        let n_b = batch_count as f64;
        let n_ab = n_a + n_b;
        self.count += batch_count;

        for i in 0..expected {
            let sum_b = batch_sum[i] as f64;
            let sum_sq_b = batch_sum_sq[i] as f64;
            let mean_b = sum_b / n_b;
            let m2_b = (sum_sq_b - sum_b * mean_b).max(0.0);

            if n_a == 0.0 {
                self.mean[i] = mean_b;
                self.m2[i] = m2_b;
            } else {
                let delta = mean_b - self.mean[i];
                self.mean[i] += delta * (n_b / n_ab);
                self.m2[i] += m2_b + delta * delta * (n_a * n_b / n_ab);
            }
        }
    }

    /// Finalizes the accumulated statistics into a [`WaveformTemplate`] (`None` if `count == 0`).
    pub fn finalize(&self) -> Option<WaveformTemplate> {
        if self.count == 0 {
            return None;
        }

        let n = self.count as f64;
        let mean: Vec<f32> = self.mean.iter().map(|&m| m as f32).collect();
        let std: Vec<f32> = self
            .m2
            .iter()
            .map(|&m2| ((m2 / n).max(0.0).sqrt()) as f32)
            .collect();

        Some(WaveformTemplate {
            num_channels: self.num_channels,
            num_samples: self.num_samples,
            mean,
            std,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::compute_mean_template;

    #[test]
    fn test_welford_matches_batch_template_exactly() {
        let snippets = vec![
            WaveformSnippet {
                primary_channel: 0,
                center_sample: 100,
                subsample_offset: 0.0,
                channel_ids: vec![0, 1],
                num_samples: 3,
                waveform: vec![-10.0, -50.0, -5.0, -2.0, -12.0, -1.0],
            },
            WaveformSnippet {
                primary_channel: 0,
                center_sample: 200,
                subsample_offset: 0.0,
                channel_ids: vec![0, 1],
                num_samples: 3,
                waveform: vec![-14.0, -60.0, -9.0, -4.0, -16.0, -3.0],
            },
            WaveformSnippet {
                primary_channel: 0,
                center_sample: 300,
                subsample_offset: 0.0,
                channel_ids: vec![0, 1],
                num_samples: 3,
                waveform: vec![-12.0, -55.0, -7.0, -3.0, -14.0, -2.0],
            },
        ];

        let batch = compute_mean_template(&snippets).unwrap();
        let mut acc = TemplateAccumulator::new(2, 3);
        for s in &snippets {
            acc.update(s);
        }
        assert_eq!(acc.count(), 3);
        let online = acc.finalize().unwrap();

        for (a, b) in online.mean.iter().zip(batch.mean.iter()) {
            assert!((a - b).abs() < 1e-5, "mean mismatch: {a} vs {b}");
        }
        for (a, b) in online.std.iter().zip(batch.std.iter()) {
            assert!((a - b).abs() < 1e-5, "std mismatch: {a} vs {b}");
        }
    }
}
