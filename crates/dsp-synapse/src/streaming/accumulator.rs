//! Online Welford accumulator for multi-channel waveform templates in $O(1)$ memory.

use dsp_base::math::RunningMoments;

use crate::core::TEMPLATE_STD_DDOF;
use crate::extraction::WaveformSnippet;
use crate::metrics::WaveformTemplate;

/// Per-channel / per-unit online accumulator of a waveform template's mean and standard deviation
/// ([`dsp_base::math::RunningMoments`]) across a stream of [`WaveformSnippet`]s and device-reduced
/// batches, without storing individual snippets.
#[derive(Debug, Clone)]
pub struct TemplateAccumulator {
    channel_ids: Vec<usize>,
    num_samples: usize,
    moments: RunningMoments,
}

impl TemplateAccumulator {
    /// Creates a new accumulator for snippets of shape `[channel_ids.len(), num_samples]` whose
    /// rows lie on recording channels `channel_ids`.
    pub fn new(channel_ids: Vec<usize>, num_samples: usize) -> Self {
        let moments = RunningMoments::new(channel_ids.len() * num_samples);
        Self { channel_ids, num_samples, moments }
    }

    /// Number of waveforms accumulated so far.
    #[inline]
    pub fn count(&self) -> u64 {
        self.moments.count()
    }

    /// Adds one [`WaveformSnippet`] (ignored when its shape differs).
    pub fn update(&mut self, snippet: &WaveformSnippet) {
        if snippet.waveform.len() == self.moments.len() {
            self.moments.push(&snippet.waveform);
        }
    }

    /// Merges a device-reduced batch of `batch_count` waveforms given their per-sample `batch_mean`
    /// and second central moment `batch_m2` (ignored when the shape differs).
    pub fn merge_batch(&mut self, batch_count: u64, batch_mean: &[f32], batch_m2: &[f32]) {
        let len = self.moments.len();
        if batch_count == 0 || batch_mean.len() != len || batch_m2.len() != len {
            return;
        }
        let mean: Vec<f64> = batch_mean.iter().map(|&v| v as f64).collect();
        let m2: Vec<f64> = batch_m2.iter().map(|&v| v as f64).collect();
        self.moments.merge(batch_count, &mean, &m2);
    }

    /// The template so far (SD with [`TEMPLATE_STD_DDOF`]; `None` before the first waveform).
    pub fn finalize(&self) -> Option<WaveformTemplate> {
        if self.count() == 0 {
            return None;
        }
        let mean = self.moments.mean().iter().map(|&m| m as f32).collect();
        let std = self.moments.std(TEMPLATE_STD_DDOF).into_iter().map(|s| s as f32).collect();
        Some(WaveformTemplate::with_count(self.channel_ids.clone(), self.num_samples, self.count() as usize, mean, std))
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
        let mut acc = TemplateAccumulator::new(vec![0, 1], 3);
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
