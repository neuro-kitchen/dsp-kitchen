use crate::core::{SpikeDetector, SpikeEvent};
use super::noise::estimate_noise_std;
use super::threshold::SpikePolarity;

/// Adaptive Exponential-Moving-Average (EMA) MAD threshold detector for non-stationary recordings
/// (e.g., respiratory bursts, posture shifts, or dynamic isometric/isotonic HD-EMG contractions).
///
/// Partitions each channel into blocks of `block_duration_ms`, updates the running noise floor
/// $\sigma_n^{(b)} = (1 - \alpha)\sigma_n^{(b-1)} + \alpha \hat{\sigma}_{\text{MAD}}^{(b)}$, and
/// detects local extrema exceeding `threshold_factor * sigma_n`.
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdDetector {
    pub threshold_factor: f32,
    pub refractory_ms: f64,
    pub block_duration_ms: f64,
    pub smoothing_alpha: f32,
    pub polarity: SpikePolarity,
}

impl Default for AdaptiveThresholdDetector {
    fn default() -> Self {
        Self {
            threshold_factor: 4.5,
            refractory_ms: 1.0,
            block_duration_ms: 100.0,
            smoothing_alpha: 0.25,
            polarity: SpikePolarity::Negative,
        }
    }
}

impl AdaptiveThresholdDetector {
    pub fn new(
        threshold_factor: f32,
        refractory_ms: f64,
        block_duration_ms: f64,
        smoothing_alpha: f32,
        polarity: SpikePolarity,
    ) -> Self {
        Self {
            threshold_factor,
            refractory_ms,
            block_duration_ms,
            smoothing_alpha: smoothing_alpha.clamp(0.01, 1.0),
            polarity,
        }
    }
}

impl SpikeDetector for AdaptiveThresholdDetector {
    fn detect(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        sample_rate_hz: f64,
    ) -> dsp_core::DspResult<Vec<SpikeEvent>> {
        assert_eq!(data.len(), channels * samples);
        let ref_samples = ((sample_rate_hz * self.refractory_ms * 1e-3).round() as usize).max(1);
        let block_samples = ((sample_rate_hz * self.block_duration_ms * 1e-3).round() as usize)
            .clamp(32, samples.max(32));
        let alpha = self.smoothing_alpha.clamp(0.01, 1.0);

        let mut all_spikes = Vec::new();

        for ch in 0..channels {
            let row = &data[ch * samples..(ch + 1) * samples];
            if samples < 3 {
                continue;
            }
            let first_end = block_samples.min(samples);
            let mut running_sigma = estimate_noise_std(&row[..first_end]).max(1e-6);
            let mut last_spike_sample: Option<usize> = None;

            let mut block_start = 0usize;
            while block_start < samples {
                let block_end = (block_start + block_samples).min(samples);
                let local_sigma = estimate_noise_std(&row[block_start..block_end]);
                if local_sigma > 0.0 {
                    running_sigma = (1.0 - alpha) * running_sigma + alpha * local_sigma;
                }
                let thresh = self.threshold_factor * running_sigma;

                let t_start = block_start.max(1);
                let t_end = block_end.min(samples.saturating_sub(1));
                for t in t_start..t_end {
                    let val = row[t];
                    let prev = row[t - 1];
                    let next = row[t + 1];

                    let is_neg = val < -thresh && val < prev && val <= next;
                    let is_pos = val > thresh && val > prev && val >= next;
                    let triggered = match self.polarity {
                        SpikePolarity::Negative => is_neg,
                        SpikePolarity::Positive => is_pos,
                        SpikePolarity::Both => is_neg || is_pos,
                    };

                    if triggered {
                        let past_refractory = match last_spike_sample {
                            None => true,
                            Some(prev_t) => t > prev_t + ref_samples,
                        };
                        if past_refractory {
                            all_spikes.push(SpikeEvent {
                                channel_id: ch,
                                sample_index: t as u64,
                                peak_amplitude_uv: val,
                            });
                            last_spike_sample = Some(t);
                        }
                    }
                }

                block_start = block_end;
            }
        }

        all_spikes.sort_by_key(|s| s.sample_index);
        Ok(all_spikes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_threshold_tracks_noise_floor_shift() {
        let fs = 10_000.0;
        let samples = 4_000;
        let mut trace = vec![0.0f32; samples];
        for i in 0..samples {
            // First half quiet (noise ~2 uV), second half noisy (noise ~10 uV)
            let scale = if i < 2_000 { 2.0 } else { 10.0 };
            trace[i] = ((i % 5) as f32 - 2.0) * scale;
        }
        // True spike in quiet regime (-25 uV) and true spike in noisy regime (-90 uV)
        trace[500] = -25.0;
        trace[3000] = -90.0;

        let detector = AdaptiveThresholdDetector::new(4.5, 1.0, 50.0, 0.5, SpikePolarity::Negative);
        let events = detector.detect(&trace, 1, samples, fs).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sample_index, 500);
        assert_eq!(events[1].sample_index, 3000);
    }
}
