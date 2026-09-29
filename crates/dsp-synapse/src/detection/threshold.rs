use serde::{Deserialize, Serialize};
use super::noise::estimate_noise_std;

/// Detected action potential event (spike).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub channel_id: usize,
    pub sample_index: u64,
    pub peak_amplitude_uv: f32,
}

/// Detects multi-channel action potential threshold crossings with adaptive noise estimation.
pub fn detect_spikes_multichannel(
    data: &[f32],
    channels: usize,
    samples: usize,
    threshold_factor: f32,
    refractory_samples: usize,
) -> Vec<SpikeEvent> {
    assert_eq!(data.len(), channels * samples);
    let mut all_spikes = Vec::new();

    for ch in 0..channels {
        let offset = ch * samples;
        let ch_slice = &data[offset..offset + samples];

        let sigma = estimate_noise_std(ch_slice);
        if sigma <= 0.0 || sigma.is_nan() {
            continue;
        }

        let thresh = -threshold_factor * sigma;
        let mut last_spike_sample = 0usize;

        for t in 1..samples.saturating_sub(1) {
            let val = ch_slice[t];
            // Negative peak detection: strictly below threshold AND lower than immediate neighbors
            if val < thresh && val < ch_slice[t - 1] && val <= ch_slice[t + 1] {
                if all_spikes.is_empty() || t > last_spike_sample + refractory_samples {
                    all_spikes.push(SpikeEvent {
                        channel_id: ch,
                        sample_index: t as u64,
                        peak_amplitude_uv: val,
                    });
                    last_spike_sample = t;
                }
            }
        }
    }

    all_spikes.sort_by_key(|s| s.sample_index);
    all_spikes
}

/// Polymorphic MAD threshold detector implementing [`crate::traits::SpikeDetector`].
#[derive(Debug, Clone)]
pub struct ThresholdSpikeDetector {
    pub threshold_factor: f32,
    pub refractory_ms: f64,
}

impl Default for ThresholdSpikeDetector {
    fn default() -> Self {
        Self {
            threshold_factor: 4.5,
            refractory_ms: 1.0,
        }
    }
}

impl crate::traits::SpikeDetector for ThresholdSpikeDetector {
    fn detect(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        sample_rate_hz: f64,
    ) -> Vec<SpikeEvent> {
        let ref_samples = ((sample_rate_hz * self.refractory_ms * 1e-3).round() as usize).max(1);
        detect_spikes_multichannel(data, channels, samples, self.threshold_factor, ref_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_estimation_and_spike_detection() {
        let mut signal = vec![0.0f32; 1000];
        for i in 0..1000 {
            signal[i] = ((i % 5) as f32 - 2.0) * 5.0;
        }
        signal[299] = -50.0;
        signal[300] = -120.0;
        signal[301] = -40.0;

        let spikes = detect_spikes_multichannel(&signal, 1, 1000, 4.0, 30);
        assert_eq!(spikes.len(), 1);
        assert_eq!(spikes[0].channel_id, 0);
        assert_eq!(spikes[0].sample_index, 300);
        assert_eq!(spikes[0].peak_amplitude_uv, -120.0);
    }
}
