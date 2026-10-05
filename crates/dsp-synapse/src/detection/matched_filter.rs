//! Prototype Matched-Filter Spike Detector (`matched_filter.rs`).
//!
//! Correlates continuous multi-channel signals against a zero-mean, unit-norm
//! canonical biphasic action potential template to boost low-SNR spikes.

use dsp_base::peaks::{local_extrema, DistanceRule, Polarity};

use crate::core::SpikeDetector;
use super::noise::estimate_noise_std;
use super::spacing::SpikeSpacing;
use super::threshold::SpikeEvent;

/// Generates a zero-mean, unit-$L_2$-norm canonical extracellular biphasic spike prototype
/// of length `kernel_samples` with negative trough at `kernel_samples / 3`.
pub fn canonical_biphasic_prototype(kernel_samples: usize) -> Vec<f32> {
    let len = kernel_samples.max(9);
    let trough_idx = (len / 3) as f32;
    let peak_idx = (2 * len / 3) as f32;
    let sigma_trough = (len as f32 / 10.0).max(1.0);
    let sigma_peak = (len as f32 / 6.0).max(1.5);

    let mut proto = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f32;
        let dt1 = (t - trough_idx) / sigma_trough;
        let dt2 = (t - peak_idx) / sigma_peak;
        let v = -1.0 * (-0.5 * dt1 * dt1).exp() + 0.38 * (-0.5 * dt2 * dt2).exp();
        proto.push(v);
    }

    // Zero-mean and unit L2 norm
    let mean: f32 = proto.iter().sum::<f32>() / (len as f32);
    for v in &mut proto {
        *v -= mean;
    }
    let norm: f32 = proto.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    for v in &mut proto {
        *v /= norm;
    }
    proto
}

/// Detects multi-channel spikes with the matched-filter inner product against `prototype`: score
/// peaks above `threshold_factor · σ(score)` where the voltage is negative, spaced by `spacing`
/// (larger score wins).
pub fn detect_spikes_matched_filter(
    data: &[f32],
    channels: usize,
    samples: usize,
    prototype: &[f32],
    threshold_factor: f32,
    spacing: SpikeSpacing,
) -> Vec<SpikeEvent> {
    assert_eq!(data.len(), channels * samples);
    let k_len = prototype.len();
    if k_len == 0 || samples <= k_len {
        return Vec::new();
    }

    // Find negative trough index in prototype so alignment matches voltage trough
    let trough_offset = prototype
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(k_len / 3);

    let mut events = Vec::new();

    for ch in 0..channels {
        let ch_off = ch * samples;
        let sig = &data[ch_off..ch_off + samples];

        let mut corr = vec![0.0f32; samples];
        for s in 0..(samples - k_len) {
            let mut dot = 0.0f32;
            let win = &sig[s..s + k_len];
            for i in 0..k_len {
                dot += win[i] * prototype[i];
            }
            corr[s + trough_offset] = dot;
        }

        let sigma_c = estimate_noise_std(&corr);
        if sigma_c <= 0.0 || sigma_c.is_nan() {
            continue;
        }

        let thresh = threshold_factor * sigma_c;
        let candidates: Vec<(usize, f32, ())> = local_extrema(&corr, Polarity::Positive)
            .into_iter()
            .filter(|&(t, _)| corr[t] >= thresh && sig[t] < 0.0)
            .map(|(t, _)| (t, corr[t], ()))
            .collect();
        events.extend(spacing.select(candidates).into_iter().map(|(t, _, ())| SpikeEvent {
            channel_id: ch,
            sample_index: t as u64,
            peak_amplitude_uv: sig[t],
        }));
    }

    events.sort_by_key(|e| (e.sample_index, e.channel_id));
    events
}

/// Polymorphic matched-filter detector implementing [`SpikeDetector`].
#[derive(Debug, Clone)]
pub struct MatchedFilterSpikeDetector {
    pub prototype: Vec<f32>,
    pub threshold_factor: f32,
    pub refractory_ms: f64,
    /// How spikes nearer than the refractory period are resolved (see [`SpikeSpacing`]).
    pub distance_rule: DistanceRule,
}

impl MatchedFilterSpikeDetector {
    pub fn new(kernel_samples: usize, threshold_factor: f32, refractory_ms: f64) -> Self {
        Self {
            prototype: canonical_biphasic_prototype(kernel_samples),
            threshold_factor,
            refractory_ms,
            distance_rule: DistanceRule::LocallyExclusive,
        }
    }
}

impl SpikeDetector for MatchedFilterSpikeDetector {
    fn detect(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        sample_rate_hz: f64,
    ) -> dsp_core::DspResult<Vec<SpikeEvent>> {
        let spacing = SpikeSpacing::from_ms(self.refractory_ms, sample_rate_hz, self.distance_rule);
        Ok(detect_spikes_matched_filter(data, channels, samples, &self.prototype, self.threshold_factor, spacing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matched_filter_spike_detection() {
        let samples = 1000;
        let mut sig = vec![0.0f32; samples];
        for i in 0..samples {
            sig[i] = ((i * 23 % 9) as f32 - 4.0) * 1.5;
        }
        let proto = canonical_biphasic_prototype(24);
        for (k, &p) in proto.iter().enumerate() {
            sig[500 + k] += p * 180.0;
        }

        let events = detect_spikes_matched_filter(&sig, 1, samples, &proto, 4.5, SpikeSpacing::new(20));
        assert_eq!(events.len(), 1);
        assert!((events[0].sample_index as i64 - 508).abs() <= 2);
    }
}
