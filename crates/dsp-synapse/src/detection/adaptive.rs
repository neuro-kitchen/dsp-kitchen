use dsp_base::peaks::{local_extrema, DistanceRule};

use crate::core::{SpikeDetector, SpikeEvent};
use super::noise::estimate_noise_std;
use super::spacing::SpikeSpacing;
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
    /// How spikes nearer than the refractory period are resolved (see [`SpikeSpacing`]).
    pub distance_rule: DistanceRule,
}

impl Default for AdaptiveThresholdDetector {
    fn default() -> Self {
        Self {
            threshold_factor: 4.5,
            refractory_ms: 1.0,
            block_duration_ms: 100.0,
            smoothing_alpha: 0.25,
            polarity: SpikePolarity::Negative,
            distance_rule: DistanceRule::LocallyExclusive,
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
            ..Self::default()
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
        let spacing = SpikeSpacing::from_ms(self.refractory_ms, sample_rate_hz, self.distance_rule);
        let block_samples = ((sample_rate_hz * self.block_duration_ms * 1e-3).round() as usize)
            .clamp(32, samples.max(32));
        let alpha = self.smoothing_alpha.clamp(0.01, 1.0);

        let mut all_spikes = Vec::new();
        for ch in 0..channels {
            let row = &data[ch * samples..(ch + 1) * samples];
            if samples < 3 {
                continue;
            }
            // Threshold of each block: σ smoothed across blocks
            let mut running_sigma = estimate_noise_std(&row[..block_samples.min(samples)]).max(1e-6);
            let thresholds: Vec<f32> = row
                .chunks(block_samples)
                .map(|block| {
                    let local_sigma = estimate_noise_std(block);
                    if local_sigma > 0.0 {
                        running_sigma = (1.0 - alpha) * running_sigma + alpha * local_sigma;
                    }
                    self.threshold_factor * running_sigma
                })
                .collect();
            let candidates: Vec<(usize, f32, ())> = local_extrema(row, self.polarity.into())
                .into_iter()
                .filter(|&(t, sign)| f32::from(sign) * row[t] >= thresholds[t / block_samples])
                .map(|(t, _)| (t, row[t].abs(), ()))
                .collect();
            all_spikes.extend(spacing.select(candidates).into_iter().map(|(t, _, ())| SpikeEvent {
                channel_id: ch,
                sample_index: t as u64,
                peak_amplitude_uv: row[t],
            }));
        }

        all_spikes.sort_by_key(|s| (s.sample_index, s.channel_id));
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
