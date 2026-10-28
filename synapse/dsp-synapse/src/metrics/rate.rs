//! Continuous Gaussian-Smoothed Instantaneous Firing Rate & Burst Detection (`rate.rs`).
//!
//! Bins discrete spike trains into a fine grid $\Delta t$ and convolves with a normalized 1D
//! Gaussian kernel ($\sigma_{\text{ms}}$, via [`dsp_base::filter::fir::gaussian_smooth_1d`]) to produce
//! a continuous firing rate estimate $r(t)$ in Hz while strictly conserving total spike count
//! ($\int_0^T r(t)\,dt \approx N_{\text{spikes}}$).

use dsp_base::filter::gaussian_smooth_1d;
use serde::{Deserialize, Serialize};

/// Continuous instantaneous firing rate profile $r(t)$ in Hz.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FiringRateCurve {
    pub time_bin_centers_sec: Vec<f64>,
    pub rate_hz: Vec<f32>,
    pub bin_width_sec: f64,
    pub kernel_sigma_ms: f64,
}

/// Detected burst epoch `[onset_sec, offset_sec]` where $r(t)$ exceeds a threshold.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BurstEpoch {
    pub onset_sec: f64,
    pub offset_sec: f64,
    pub peak_rate_hz: f32,
}

/// Computes a continuous Gaussian-smoothed instantaneous firing rate $r(t)$ (in Hz) over `[0, total_duration_sec]`.
pub fn compute_instantaneous_firing_rate(
    spike_times_sec: &[f64],
    total_duration_sec: f64,
    bin_width_ms: f64,
    kernel_sigma_ms: f64,
) -> FiringRateCurve {
    let dt = (bin_width_ms * 1e-3).max(1e-4);
    let num_bins = ((total_duration_sec.max(dt) / dt).ceil() as usize).max(1);
    let mut raw_rate = vec![0.0f32; num_bins];
    let inv_dt = (1.0 / dt) as f32;

    for &t in spike_times_sec {
        if t >= 0.0 && t <= total_duration_sec {
            let b = ((t / dt).floor() as usize).min(num_bins - 1);
            raw_rate[b] += inv_dt;
        }
    }

    let sigma_samples = (kernel_sigma_ms.max(0.0) / bin_width_ms.max(1e-3)) as f32;
    let rate_hz = if sigma_samples > 0.05 {
        gaussian_smooth_1d(&raw_rate, sigma_samples)
    } else {
        raw_rate
    };

    let time_bin_centers_sec: Vec<f64> = (0..num_bins).map(|b| (b as f64 + 0.5) * dt).collect();
    FiringRateCurve {
        time_bin_centers_sec,
        rate_hz,
        bin_width_sec: dt,
        kernel_sigma_ms,
    }
}

/// Detects burst epochs where the continuous firing rate exceeds `threshold_hz` for at least `min_duration_ms`.
pub fn detect_burst_epochs(
    curve: &FiringRateCurve,
    threshold_hz: f32,
    min_duration_ms: f64,
) -> Vec<BurstEpoch> {
    let mut epochs = Vec::new();
    let min_dur_sec = min_duration_ms * 1e-3;
    let dt = curve.bin_width_sec;

    let mut in_burst = false;
    let mut start_idx = 0usize;
    let mut peak_rate = 0.0f32;

    for (i, &r) in curve.rate_hz.iter().enumerate() {
        if r >= threshold_hz {
            if !in_burst {
                in_burst = true;
                start_idx = i;
                peak_rate = r;
            } else if r > peak_rate {
                peak_rate = r;
            }
        } else if in_burst {
            let onset = start_idx as f64 * dt;
            let offset = i as f64 * dt;
            if offset - onset >= min_dur_sec {
                epochs.push(BurstEpoch {
                    onset_sec: onset,
                    offset_sec: offset,
                    peak_rate_hz: peak_rate,
                });
            }
            in_burst = false;
        }
    }

    if in_burst {
        let onset = start_idx as f64 * dt;
        let offset = curve.rate_hz.len() as f64 * dt;
        if offset - onset >= min_dur_sec {
            epochs.push(BurstEpoch {
                onset_sec: onset,
                offset_sec: offset,
                peak_rate_hz: peak_rate,
            });
        }
    }

    epochs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_firing_rate_conserves_spike_count_and_detects_burst() {
        // 50 spikes clustered in a burst around t = 1.0..1.5 s in a 3.0 s recording
        let spikes: Vec<f64> = (0..50).map(|i| 1.0 + (i as f64) * 0.01).collect();
        let curve = compute_instantaneous_firing_rate(&spikes, 3.0, 5.0, 20.0);

        let integral_spikes: f64 = curve
            .rate_hz
            .iter()
            .map(|&r| (r as f64) * curve.bin_width_sec)
            .sum();
        assert!(
            (integral_spikes - 50.0).abs() < 0.5,
            "integral={integral_spikes}, expected 50.0"
        );

        let bursts = detect_burst_epochs(&curve, 40.0, 100.0);
        assert_eq!(bursts.len(), 1);
        assert!((bursts[0].onset_sec - 1.0).abs() < 0.1);
        assert!((bursts[0].offset_sec - 1.5).abs() < 0.1);
        assert!(bursts[0].peak_rate_hz > 80.0);
    }
}
