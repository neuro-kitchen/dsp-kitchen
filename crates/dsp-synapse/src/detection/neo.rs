//! Teager-Kaiser Nonlinear Energy Operator (NEO) spike detection (`neo.rs`).
//!
//! Computes the discrete Teager-Kaiser energy $\psi[n] = x^2[n] - x[n-1]x[n+1]$,
//! which simultaneously boosts high-frequency, high-amplitude action potentials
//! while suppressing low-frequency LFP hum.

use crate::traits::SpikeDetector;
use super::noise::estimate_noise_std;
use super::threshold::SpikeEvent;

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

/// Detects multi-channel spikes using the Teager-Kaiser Nonlinear Energy Operator (NEO).
pub fn detect_spikes_neo(
    data: &[f32],
    channels: usize,
    samples: usize,
    neo_threshold_factor: f32,
    refractory_samples: usize,
) -> Vec<SpikeEvent> {
    assert_eq!(data.len(), channels * samples);
    let mut events = Vec::new();
    if samples < 3 {
        return events;
    }

    for ch in 0..channels {
        let off = ch * samples;
        let ch_slice = &data[off..off + samples];
        let psi = compute_neo_energy_1d(ch_slice);

        // Estimate baseline NEO energy via MAD / median
        let neo_std = estimate_noise_std(&psi);
        if neo_std <= 0.0 || neo_std.is_nan() {
            continue;
        }
        let thresh = neo_threshold_factor * neo_std;
        let mut last_spike = 0usize;

        for t in 1..(samples - 1) {
            let e = psi[t];
            if e > thresh && e >= psi[t - 1] && e >= psi[t + 1] && ch_slice[t] < 0.0 {
                if events.is_empty() || t > last_spike + refractory_samples {
                    // Refine to local negative voltage trough within +/- 2 samples
                    let w_start = t.saturating_sub(2);
                    let w_end = (t + 2).min(samples - 1);
                    let mut best_t = t;
                    let mut min_v = ch_slice[t];
                    for k in w_start..=w_end {
                        if ch_slice[k] < min_v {
                            min_v = ch_slice[k];
                            best_t = k;
                        }
                    }

                    events.push(SpikeEvent {
                        channel_id: ch,
                        sample_index: best_t as u64,
                        peak_amplitude_uv: min_v,
                    });
                    last_spike = best_t;
                }
            }
        }
    }

    events.sort_by_key(|s| s.sample_index);
    events
}

/// Polymorphic wrapper for Teager-Kaiser NEO detection implementing [`SpikeDetector`].
#[derive(Debug, Clone)]
pub struct NeoSpikeDetector {
    pub threshold_factor: f32,
    pub refractory_ms: f64,
}

impl Default for NeoSpikeDetector {
    fn default() -> Self {
        Self {
            threshold_factor: 8.0,
            refractory_ms: 1.0,
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
    ) -> Vec<SpikeEvent> {
        let ref_samples = ((sample_rate_hz * self.refractory_ms * 1e-3).round() as usize).max(1);
        detect_spikes_neo(data, channels, samples, self.threshold_factor, ref_samples)
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

        let spikes = detect_spikes_neo(&sig, 1, samples, 8.0, 20);
        assert_eq!(spikes.len(), 1);
        assert_eq!(spikes[0].sample_index, 400);
        assert_eq!(spikes[0].peak_amplitude_uv, -95.0);
    }
}
