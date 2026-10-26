//! Muscle Fiber Conduction Velocity (MFCV) estimation for planar HD-EMG grids (`conduction.rs`).
//!
//! Estimates action potential propagation speed $v$ (physiologically 2–7 m/s) along longitudinal
//! muscle fiber columns of a 2D HD-EMG array using sub-sample parabolic cross-correlation between
//! adjacent single-differential channels.

use serde::{Deserialize, Serialize};
use dsp_base::math::{cross_correlation, peak_lag};

/// Largest delay searched between adjacent channels, as a fraction of the snippet length.
const MAX_LAG_FRACTION: usize = 3;
/// Channel pairs with less energy product than this are skipped (flat signals).
const MIN_PAIR_ENERGY: f32 = 1e-8;
/// Normalized correlation a channel pair needs to count.
const MIN_PAIR_CORRELATION: f32 = 0.2;
/// Delays below this (samples) carry no direction and are skipped.
const MIN_DELAY_SAMPLES: f32 = 1e-3;

impl ConductionVelocityEstimate {
    /// No estimate (too few channels or samples, or no correlated pair): every field NaN.
    pub const UNDEFINED: Self = Self { velocity_m_per_s: f32::NAN, mean_delay_ms: f32::NAN, correlation_r: f32::NAN };
}

/// Estimated Muscle Fiber Conduction Velocity (MFCV) along a longitudinal HD-EMG electrode column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConductionVelocityEstimate {
    /// Propagation speed in meters per second ($\text{m/s}$, physiological human range $3.0 - 6.0\,\text{m/s}$).
    pub velocity_m_per_s: f32,
    /// Mean inter-electrode delay in milliseconds ($\text{ms}$).
    pub mean_delay_ms: f32,
    /// Normalized peak cross-correlation coefficient $r \in [0, 1]$ across longitudinal channel pairs.
    pub correlation_r: f32,
}

/// Estimates Muscle Fiber Conduction Velocity (MFCV) from a multi-channel HD-EMG template or snippet
/// (`[channels, samples]`, organized as `rows x cols` where `cols` runs along the muscle fiber axis)
/// with inter-electrode spacing `pitch_um` ($\mu\text{m}$) and `sample_rate_hz`.
pub fn estimate_hdemg_conduction_velocity(
    waveform: &[f32],
    rows: usize,
    cols: usize,
    samples: usize,
    pitch_um: f32,
    sample_rate_hz: f64,
) -> ConductionVelocityEstimate {
    assert_eq!(waveform.len(), rows * cols * samples);
    if rows == 0 || cols < 3 || samples < 5 {
        return ConductionVelocityEstimate::UNDEFINED;
    }

    // 1. Find the row with maximal RMS energy
    let mut best_row = 0usize;
    let mut best_energy = -1.0f32;
    for r in 0..rows {
        let mut e = 0.0f32;
        for c in 0..cols {
            let ch = r * cols + c;
            let slice = &waveform[ch * samples..(ch + 1) * samples];
            e += slice.iter().map(|v| v * v).sum::<f32>();
        }
        if e > best_energy {
            best_energy = e;
            best_row = r;
        }
    }

    // 2. Compute longitudinal single-differential signals along `best_row` (cols - 1 pairs)
    let num_sd = cols - 1;
    let mut sd = vec![0.0f32; num_sd * samples];
    for c in 0..num_sd {
        let ch0 = best_row * cols + c;
        let ch1 = best_row * cols + c + 1;
        for t in 0..samples {
            sd[c * samples + t] = waveform[ch1 * samples + t] - waveform[ch0 * samples + t];
        }
    }

    // 3. Cross-correlate adjacent single-differential channels with sub-sample parabolic refinement
    let max_lag = (samples / MAX_LAG_FRACTION).max(2);
    let mut weighted_delay_samples = 0.0f32;
    let mut weight_sum = 0.0f32;
    let mut corr_sum = 0.0f32;
    let mut pair_count = 0usize;

    for c in 0..(num_sd - 1) {
        let x = &sd[c * samples..(c + 1) * samples];
        let y = &sd[(c + 1) * samples..(c + 2) * samples];
        let ex: f32 = x.iter().map(|v| v * v).sum();
        let ey: f32 = y.iter().map(|v| v * v).sum();
        let denom = (ex * ey).sqrt();
        if denom < MIN_PAIR_ENERGY {
            continue;
        }

        let corrs: Vec<f32> = cross_correlation(x, y, max_lag).into_iter().map(|c| c / denom).collect();
        let Some(peak) = peak_lag(&corrs, max_lag) else { continue };
        let best_r = peak.value;
        let lag_samples = peak.fractional_lag().abs();
        if best_r > MIN_PAIR_CORRELATION && lag_samples > MIN_DELAY_SAMPLES {
            let w = best_r * denom;
            weighted_delay_samples += w * lag_samples;
            weight_sum += w;
            corr_sum += best_r;
            pair_count += 1;
        }
    }

    if weight_sum <= MIN_PAIR_ENERGY || pair_count == 0 {
        return ConductionVelocityEstimate::UNDEFINED;
    }

    let mean_lag_samples = weighted_delay_samples / weight_sum;
    let delay_sec = (mean_lag_samples as f64) / sample_rate_hz.max(1.0);
    let pitch_m = (pitch_um as f64) * 1e-6;
    // Delays are at least MIN_DELAY_SAMPLES, so the velocity is finite
    let velocity_m_per_s = (pitch_m / delay_sec) as f32;

    ConductionVelocityEstimate {
        velocity_m_per_s,
        mean_delay_ms: (delay_sec * 1e3) as f32,
        correlation_r: corr_sum / (pair_count as f32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdemg_conduction_velocity_estimation() {
        // 4x8 grid, 8000 um (8 mm) inter-electrode pitch, 4000 Hz sample rate
        // True velocity = 4.0 m/s -> delay per 8 mm pitch = 2.0 ms = 8.0 samples
        let (rows, cols, samples) = (4, 8, 96);
        let pitch_um = 8000.0f32;
        let fs = 4000.0f64;
        let mut wave = vec![0.0f32; rows * cols * samples];

        for c in 0..cols {
            let ch = 1 * cols + c;
            let center = 20.0 + (c as f32) * 8.0;
            for t in 0..samples {
                let u = (t as f32 - center) / 3.5;
                // Biphasic propagating action potential
                wave[ch * samples + t] = -100.0 * u * (-0.5 * u * u).exp() * (c as f32 * 0.05 + 1.0);
            }
        }

        let est = estimate_hdemg_conduction_velocity(&wave, rows, cols, samples, pitch_um, fs);
        assert!(
            (est.velocity_m_per_s - 4.0).abs() < 0.3,
            "est velocity = {} m/s, expected 4.0 m/s",
            est.velocity_m_per_s
        );
        assert!(est.correlation_r > 0.9);
    }
}
