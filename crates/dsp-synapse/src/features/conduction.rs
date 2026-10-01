//! Muscle Fiber Conduction Velocity (MFCV) estimation for planar HD-EMG grids (`conduction.rs`).
//!
//! Estimates action potential propagation speed $v \in [2.0, 7.0]\,\text{m/s}$ along longitudinal
//! muscle fiber columns of a 2D HD-EMG array using sub-sample parabolic cross-correlation between
//! adjacent single-differential channels.

use serde::{Deserialize, Serialize};
use crate::extraction::parabolic_subsample_offset;

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
        return ConductionVelocityEstimate {
            velocity_m_per_s: 0.0,
            mean_delay_ms: 0.0,
            correlation_r: 0.0,
        };
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
    let max_lag = (samples / 3).max(2) as isize;
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
        if denom < 1e-8 {
            continue;
        }

        let n_lags = (2 * max_lag + 1) as usize;
        let mut corrs = vec![0.0f32; n_lags];
        let mut best_idx = max_lag as usize;
        let mut best_r = f32::NEG_INFINITY;

        for (idx, lag) in (-max_lag..=max_lag).enumerate() {
            let mut dot = 0.0f32;
            for t in 0..samples {
                let shifted = t as isize + lag;
                if shifted >= 0 && (shifted as usize) < samples {
                    dot += x[t] * y[shifted as usize];
                }
            }
            let r = dot / denom;
            corrs[idx] = r;
            if r > best_r {
                best_r = r;
                best_idx = idx;
            }
        }

        let sub = if best_idx > 0 && best_idx + 1 < n_lags {
            parabolic_subsample_offset(-corrs[best_idx - 1], -corrs[best_idx], -corrs[best_idx + 1])
        } else {
            0.0
        };
        let lag_samples = ((best_idx as isize - max_lag) as f32 + sub).abs();
        if best_r > 0.2 && lag_samples > 1e-3 {
            let w = best_r * denom;
            weighted_delay_samples += w * lag_samples;
            weight_sum += w;
            corr_sum += best_r;
            pair_count += 1;
        }
    }

    if weight_sum <= 1e-8 || pair_count == 0 {
        return ConductionVelocityEstimate {
            velocity_m_per_s: 0.0,
            mean_delay_ms: 0.0,
            correlation_r: 0.0,
        };
    }

    let mean_lag_samples = weighted_delay_samples / weight_sum;
    let delay_sec = (mean_lag_samples as f64) / sample_rate_hz.max(1.0);
    let pitch_m = (pitch_um as f64) * 1e-6;
    let velocity_m_per_s = if delay_sec > 1e-9 {
        (pitch_m / delay_sec) as f32
    } else {
        0.0
    };

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
