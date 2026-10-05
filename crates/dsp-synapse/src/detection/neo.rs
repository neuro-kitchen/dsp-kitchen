//! Teager-Kaiser Nonlinear Energy Operator (NEO) spike detection (`neo.rs`).
//!
//! Computes the discrete Teager-Kaiser energy $\psi[n] = x^2[n] - x[n-1]x[n+1]$,
//! which simultaneously boosts high-frequency, high-amplitude action potentials
//! while suppressing low-frequency LFP hum.

use dsp_base::peaks::{local_extrema, DistanceRule, Polarity};

use crate::core::SpikeDetector;
use super::noise::estimate_noise_std;
use super::spacing::SpikeSpacing;
use super::threshold::SpikeEvent;

/// Samples on each side of an energy peak searched for the voltage trough.
const TROUGH_SEARCH_SAMPLES: usize = 2;

/// Computes the 1D Teager-Kaiser Nonlinear Energy Operator $\psi[n] = x^2[n] - x[n-1]x[n+1]$.
pub fn compute_neo_energy_1d(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let mut psi = vec![0.0f32; n];
    if n < 3 {
        return psi;
    }
    for i in 1..(n - 1) {
        let x0 = signal[i];
        psi[i] = (x0 * x0 - signal[i - 1] * signal[i + 1]).max(0.0);
    }
    psi[0] = psi[1];
    psi[n - 1] = psi[n - 2];
    psi
}

/// Detects multi-channel spikes with the Teager-Kaiser Nonlinear Energy Operator (NEO): energy
/// peaks above `neo_threshold_factor · σ(ψ)` where the voltage is negative, moved to the voltage
/// trough within ±2 samples, spaced by `spacing` (larger energy wins).
pub fn detect_spikes_neo(
    data: &[f32],
    channels: usize,
    samples: usize,
    neo_threshold_factor: f32,
    spacing: SpikeSpacing,
) -> Vec<SpikeEvent> {
    assert_eq!(data.len(), channels * samples);
    let mut events = Vec::new();
    if samples < 3 {
        return events;
    }

    for ch in 0..channels {
        let ch_slice = &data[ch * samples..(ch + 1) * samples];
        let psi = compute_neo_energy_1d(ch_slice);

        // Baseline NEO energy via MAD / median
        let neo_std = estimate_noise_std(&psi);
        if neo_std <= 0.0 || neo_std.is_nan() {
            continue;
        }
        let thresh = neo_threshold_factor * neo_std;

        let candidates: Vec<(usize, f32, f32)> = local_extrema(&psi, Polarity::Positive)
            .into_iter()
            .filter(|&(t, _)| psi[t] >= thresh && ch_slice[t] < 0.0)
            .map(|(t, _)| {
                let window = t.saturating_sub(TROUGH_SEARCH_SAMPLES)..=(t + TROUGH_SEARCH_SAMPLES).min(samples - 1);
                let trough = window.min_by(|&a, &b| ch_slice[a].total_cmp(&ch_slice[b])).unwrap_or(t);
                (trough, psi[t], ch_slice[trough])
            })
            .collect();
        events.extend(spacing.select(candidates).into_iter().map(|(t, _, v)| SpikeEvent {
            channel_id: ch,
            sample_index: t as u64,
            peak_amplitude_uv: v,
        }));
    }

    events.sort_by_key(|s| (s.sample_index, s.channel_id));
    events
}

/// Polymorphic wrapper for Teager-Kaiser NEO detection implementing [`SpikeDetector`].
#[derive(Debug, Clone)]
pub struct NeoSpikeDetector {
    pub threshold_factor: f32,
    pub refractory_ms: f64,
    /// How spikes nearer than the refractory period are resolved (see [`SpikeSpacing`]).
    pub distance_rule: DistanceRule,
}

impl Default for NeoSpikeDetector {
    fn default() -> Self {
        Self {
            threshold_factor: 8.0,
            refractory_ms: 1.0,
            distance_rule: DistanceRule::LocallyExclusive,
        }
    }
}

impl SpikeDetector for NeoSpikeDetector {
    fn detect(
        &self,
        data: &[f32],
        channels: usize,
        samples: usize,
        sample_rate_hz: f64,
    ) -> dsp_core::DspResult<Vec<SpikeEvent>> {
        let spacing = SpikeSpacing::from_ms(self.refractory_ms, sample_rate_hz, self.distance_rule);
        Ok(detect_spikes_neo(data, channels, samples, self.threshold_factor, spacing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neo_spike_detection() {
        let samples = 1000;
        let mut sig = vec![0.0f32; samples];
        for i in 0..samples {
            sig[i] = ((i * 13 % 11) as f32 - 5.0) * 1.5;
        }
        // Sharp action potential at sample 400
        sig[399] = 20.0;
        sig[400] = -95.0;
        sig[401] = 35.0;

        let spikes = detect_spikes_neo(&sig, 1, samples, 8.0, SpikeSpacing::new(20));
        assert_eq!(spikes.len(), 1);
        assert_eq!(spikes[0].sample_index, 400);
        assert_eq!(spikes[0].peak_amplitude_uv, -95.0);
    }
}
