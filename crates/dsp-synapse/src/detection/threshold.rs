pub use crate::core::SpikeEvent;
use super::noise::estimate_noise_std;
use serde::{Deserialize, Serialize};

/// Polarity mode for action potential peak detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpikePolarity {
    /// Extracellular negative troughs ($x[t] < -\alpha \sigma_n$).
    #[default]
    Negative,
    /// Positive peaks ($x[t] > +\alpha \sigma_n$, e.g. axonal return currents or rectified EMG).
    Positive,
    /// Biphasic / dual-polarity local extrema ($|x[t]| > \alpha \sigma_n$).
    Both,
}

/// Detects multi-channel negative action potential threshold crossings with adaptive noise estimation.
pub fn detect_spikes_multichannel(
    data: &[f32],
    channels: usize,
    samples: usize,
    threshold_factor: f32,
    refractory_samples: usize,
) -> Vec<SpikeEvent> {
    detect_spikes_multichannel_polarity(
        data,
        channels,
        samples,
        threshold_factor,
        refractory_samples,
        SpikePolarity::Negative,
    )
}

/// Detects multi-channel action potential threshold crossings with configurable [`SpikePolarity`].
pub fn detect_spikes_multichannel_polarity(
    data: &[f32],
    channels: usize,
    samples: usize,
    threshold_factor: f32,
    refractory_samples: usize,
    polarity: SpikePolarity,
) -> Vec<SpikeEvent> {
    assert_eq!(data.len(), channels * samples);
    let sigmas: Vec<f32> = (0..channels)
        .map(|ch| {
            let offset = ch * samples;
            estimate_noise_std(&data[offset..offset + samples])
        })
        .collect();
    detect_spikes_with_sigma_polarity(
        data,
        channels,
        samples,
        &sigmas,
        threshold_factor,
        refractory_samples,
        polarity,
    )
}

/// Detects multi-channel negative action potential threshold crossings using pre-calibrated per-channel
/// noise standard deviations `channel_sigmas` ($\sigma_n$ in $\mu\text{V}$).
pub fn detect_spikes_with_sigma(
    data: &[f32],
    channels: usize,
    samples: usize,
    channel_sigmas: &[f32],
    threshold_factor: f32,
    refractory_samples: usize,
) -> Vec<SpikeEvent> {
    detect_spikes_with_sigma_polarity(
        data,
        channels,
        samples,
        channel_sigmas,
        threshold_factor,
        refractory_samples,
        SpikePolarity::Negative,
    )
}

/// Detects multi-channel action potential threshold crossings using pre-calibrated per-channel
/// noise standard deviations `channel_sigmas` ($\sigma_n$ in $\mu\text{V}$) and [`SpikePolarity`].
pub fn detect_spikes_with_sigma_polarity(
    data: &[f32],
    channels: usize,
    samples: usize,
    channel_sigmas: &[f32],
    threshold_factor: f32,
    refractory_samples: usize,
    polarity: SpikePolarity,
) -> Vec<SpikeEvent> {
    assert_eq!(data.len(), channels * samples);
    assert_eq!(channel_sigmas.len(), channels);
    let mut all_spikes = Vec::new();

    for (ch, &sigma) in channel_sigmas.iter().enumerate() {
        if sigma <= 0.0 || sigma.is_nan() {
            continue;
        }

        let offset = ch * samples;
        let ch_slice = &data[offset..offset + samples];
        let pos_thresh = threshold_factor * sigma;
        let neg_thresh = -pos_thresh;
        let mut last_spike_sample: Option<usize> = None;

        for t in 1..samples.saturating_sub(1) {
            let val = ch_slice[t];
            let prev = ch_slice[t - 1];
            let next = ch_slice[t + 1];

            let is_neg_peak = val < neg_thresh && val < prev && val <= next;
            let is_pos_peak = val > pos_thresh && val > prev && val >= next;

            let triggered = match polarity {
                SpikePolarity::Negative => is_neg_peak,
                SpikePolarity::Positive => is_pos_peak,
                SpikePolarity::Both => is_neg_peak || is_pos_peak,
            };

            if triggered {
                let past_refractory = match last_spike_sample {
                    None => true,
                    Some(prev_t) => t > prev_t + refractory_samples,
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
    }

    all_spikes.sort_by_key(|s| s.sample_index);
    all_spikes
}

/// Polymorphic MAD threshold detector implementing [`crate::core::SpikeDetector`].
#[derive(Debug, Clone)]
pub struct ThresholdSpikeDetector {
    pub threshold_factor: f32,
    pub refractory_ms: f64,
    pub polarity: SpikePolarity,
}

impl Default for ThresholdSpikeDetector {
    fn default() -> Self {
        Self {
            threshold_factor: 4.5,
            refractory_ms: 1.0,
            polarity: SpikePolarity::Negative,
        }
    }
}

impl ThresholdSpikeDetector {
    pub fn new(threshold_factor: f32, refractory_ms: f64, polarity: SpikePolarity) -> Self {
        Self {
            threshold_factor,
            refractory_ms,
            polarity,
        }
    }
}

impl crate::core::SpikeDetector for ThresholdSpikeDetector {
    fn detect(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        sample_rate_hz: f64,
    ) -> dsp_core::DspResult<Vec<SpikeEvent>> {
        let ref_samples = ((sample_rate_hz * self.refractory_ms * 1e-3).round() as usize).max(1);
        Ok(detect_spikes_multichannel_polarity(
            data,
            channels,
            samples,
            self.threshold_factor,
            ref_samples,
            self.polarity,
        ))
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

    #[test]
    fn test_positive_and_both_polarity_detection() {
        let mut signal = vec![0.0f32; 1000];
        for i in 0..1000 {
            signal[i] = ((i % 5) as f32 - 2.0) * 3.0;
        }
        signal[200] = -95.0;
        signal[600] = 110.0;

        let pos = detect_spikes_multichannel_polarity(&signal, 1, 1000, 4.5, 20, SpikePolarity::Positive);
        assert_eq!(pos.len(), 1);
        assert_eq!(pos[0].sample_index, 600);
        assert_eq!(pos[0].peak_amplitude_uv, 110.0);

        let both = detect_spikes_multichannel_polarity(&signal, 1, 1000, 4.5, 20, SpikePolarity::Both);
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].sample_index, 200);
        assert_eq!(both[1].sample_index, 600);
    }
}
