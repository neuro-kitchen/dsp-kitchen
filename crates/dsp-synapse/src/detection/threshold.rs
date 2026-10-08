pub use crate::core::SpikeEvent;
use dsp_base::peaks::{find_peaks, DistanceRule, Interval, PeakOptions};
use serde::{Deserialize, Serialize};

use super::noise::estimate_noise_std;
use super::spacing::SpikeSpacing;

/// Polarity mode for action potential peak detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpikePolarity {
    /// Extracellular negative troughs (`x[t] < −α·σₙ`).
    #[default]
    Negative,
    /// Positive peaks (`x[t] > +α·σₙ`, e.g. axonal return currents or rectified EMG).
    Positive,
    /// Biphasic / dual-polarity local extrema (`|x[t]| > α·σₙ`).
    Both,
}

/// Detection height per channel: `threshold_factor · σ`, `+∞` (no detection) where `σ` is not a
/// positive number.
pub fn detection_heights(channel_sigmas: &[f32], threshold_factor: f32) -> Vec<f32> {
    channel_sigmas
        .iter()
        .map(|&sigma| if sigma > 0.0 { threshold_factor * sigma } else { f32::INFINITY })
        .collect()
}

/// Threshold crossings of every channel of `[channels, samples]` `data`: local extrema of
/// `polarity` reaching `threshold_factor · σ` (σ per channel by [`estimate_noise_std`]), spaced by
/// `spacing`; sorted by sample.
pub fn detect_spikes_multichannel(
    data: &[f32],
    channels: usize,
    samples: usize,
    threshold_factor: f32,
    polarity: SpikePolarity,
    spacing: SpikeSpacing,
) -> Vec<SpikeEvent> {
    assert_eq!(data.len(), channels * samples);
    let sigmas: Vec<f32> = data.chunks_exact(samples.max(1)).take(channels).map(estimate_noise_std).collect();
    detect_spikes_with_sigma(data, channels, samples, &sigmas, threshold_factor, polarity, spacing)
}

/// [`detect_spikes_multichannel`] with pre-calibrated per-channel noise `channel_sigmas` (µV).
pub fn detect_spikes_with_sigma(
    data: &[f32],
    channels: usize,
    samples: usize,
    channel_sigmas: &[f32],
    threshold_factor: f32,
    polarity: SpikePolarity,
    spacing: SpikeSpacing,
) -> Vec<SpikeEvent> {
    assert_eq!(data.len(), channels * samples);
    assert_eq!(channel_sigmas.len(), channels);
    let heights = detection_heights(channel_sigmas, threshold_factor);
    let mut all_spikes = Vec::new();
    for (ch, &height) in heights.iter().enumerate() {
        if !height.is_finite() {
            continue;
        }
        let row = &data[ch * samples..(ch + 1) * samples];
        let options = PeakOptions {
            height: Interval::at_least(height),
            distance: Some(spacing.distance()),
            distance_rule: spacing.rule,
            ..Default::default()
        };
        let peaks = find_peaks(row, polarity.into(), &options);
        all_spikes.extend(peaks.indices.into_iter().map(|t| SpikeEvent { channel_id: ch, sample_index: t as u64, peak_amplitude_uv: row[t] }));
    }
    all_spikes.sort_by_key(|s| (s.sample_index, s.channel_id));
    all_spikes
}

/// Polymorphic MAD threshold detector implementing [`crate::core::SpikeDetector`].
#[derive(Debug, Clone)]
pub struct ThresholdSpikeDetector {
    pub threshold_factor: f32,
    pub refractory_ms: f64,
    pub polarity: SpikePolarity,
    /// How spikes nearer than the refractory period are resolved (see [`SpikeSpacing`]).
    pub distance_rule: DistanceRule,
}

impl Default for ThresholdSpikeDetector {
    fn default() -> Self {
        Self {
            threshold_factor: 4.5,
            refractory_ms: 1.0,
            polarity: SpikePolarity::Negative,
            distance_rule: DistanceRule::LocallyExclusive,
        }
    }
}

impl ThresholdSpikeDetector {
    pub fn new(threshold_factor: f32, refractory_ms: f64, polarity: SpikePolarity) -> Self {
        Self { threshold_factor, refractory_ms, polarity, ..Self::default() }
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
        let spacing = SpikeSpacing::from_ms(self.refractory_ms, sample_rate_hz, self.distance_rule);
        Ok(detect_spikes_multichannel(data, channels, samples, self.threshold_factor, self.polarity, spacing))
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

        let spikes = detect_spikes_multichannel(&signal, 1, 1000, 4.0, SpikePolarity::Negative, SpikeSpacing::new(30));
        assert_eq!(spikes.len(), 1);
        assert_eq!(spikes[0].channel_id, 0);
        assert_eq!(spikes[0].sample_index, 300);
        assert_eq!(spikes[0].peak_amplitude_uv, -120.0);
    }

    #[test]
    fn test_larger_spike_wins_within_refractory() {
        // A small early crossing must not hide the real spike 5 samples later
        let mut signal: Vec<f32> = (0..1000).map(|i| ((i % 5) as f32 - 2.0) * 3.0).collect();
        signal[400] = -60.0;
        signal[405] = -150.0;
        let spikes = detect_spikes_multichannel(&signal, 1, 1000, 4.5, SpikePolarity::Negative, SpikeSpacing::new(20));
        assert_eq!(spikes.iter().map(|s| s.sample_index).collect::<Vec<_>>(), vec![405]);
    }

    #[test]
    fn test_positive_and_both_polarity_detection() {
        let mut signal = vec![0.0f32; 1000];
        for i in 0..1000 {
            signal[i] = ((i % 5) as f32 - 2.0) * 3.0;
        }
        signal[200] = -95.0;
        signal[600] = 110.0;

        let pos = detect_spikes_multichannel(&signal, 1, 1000, 4.5, SpikePolarity::Positive, SpikeSpacing::new(20));
        assert_eq!(pos.len(), 1);
        assert_eq!(pos[0].sample_index, 600);
        assert_eq!(pos[0].peak_amplitude_uv, 110.0);

        let both = detect_spikes_multichannel(&signal, 1, 1000, 4.5, SpikePolarity::Both, SpikeSpacing::new(20));
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].sample_index, 200);
        assert_eq!(both[1].sample_index, 600);
    }
}
