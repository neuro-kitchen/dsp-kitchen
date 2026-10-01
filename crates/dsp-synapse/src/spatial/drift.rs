//! 2D Depth-Time Activity Histogram & Cross-Correlation Probe Drift Estimation (`drift_map.rs`).
//!
//! Tracks vertical mechanical micromotion $d(t)$ of high-density probes (e.g., Neuropixels)
//! by binning spike depths and log-amplitudes into a 2D spatiotemporal histogram and
//! registering each temporal column via sub-bin parabolic cross-correlation.

use serde::{Deserialize, Serialize};
use crate::extraction::parabolic_subsample_offset;

/// Estimated vertical probe drift trace and 2D activity histogram.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriftEstimate {
    pub time_bin_centers_sec: Vec<f64>,
    /// Estimated vertical displacement $d(t)$ in $\mu\text{m}$ at each time bin.
    pub drift_um: Vec<f32>,
    /// Flat 2D histogram of shape `[num_time_bins, num_depth_bins]`.
    pub activity_map: Vec<f32>,
    pub num_time_bins: usize,
    pub num_depth_bins: usize,
    pub depth_min_um: f32,
    pub depth_bin_size_um: f32,
}

impl DriftEstimate {
    /// Linearly interpolates the estimated vertical drift $d(t)$ at arbitrary timestamp `t_sec`.
    pub fn interpolate_drift_at(&self, t_sec: f64) -> f32 {
        let n = self.time_bin_centers_sec.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 || t_sec <= self.time_bin_centers_sec[0] {
            return self.drift_um[0];
        }
        if t_sec >= self.time_bin_centers_sec[n - 1] {
            return self.drift_um[n - 1];
        }

        let pos = self
            .time_bin_centers_sec
            .partition_point(|&tc| tc < t_sec)
            .clamp(1, n - 1);
        let t0 = self.time_bin_centers_sec[pos - 1];
        let t1 = self.time_bin_centers_sec[pos];
        let alpha = ((t_sec - t0) / (t1 - t0).max(1e-9)) as f32;
        self.drift_um[pos - 1] * (1.0 - alpha) + self.drift_um[pos] * alpha
    }
}

/// Estimates rigid vertical probe drift $d(t)$ from spike timestamps, depths (`y_um`), and amplitudes.
pub fn estimate_rigid_drift(
    spike_times_sec: &[f64],
    spike_depths_um: &[f32],
    spike_amplitudes_uv: &[f32],
    total_duration_sec: f64,
    time_bin_sec: f64,
    depth_min_um: f32,
    depth_max_um: f32,
    depth_bin_size_um: f32,
    max_drift_um: f32,
) -> DriftEstimate {
    let dt = time_bin_sec.max(0.1);
    let dz = depth_bin_size_um.max(1.0);
    let num_time_bins = ((total_duration_sec / dt).ceil() as usize).max(1);
    let num_depth_bins = (((depth_max_um - depth_min_um).max(dz) / dz).ceil() as usize).max(4);

    let mut activity_map = vec![0.0f32; num_time_bins * num_depth_bins];
    let n_spikes = spike_times_sec
        .len()
        .min(spike_depths_um.len())
        .min(spike_amplitudes_uv.len());

    for i in 0..n_spikes {
        let t = spike_times_sec[i];
        let z = spike_depths_um[i];
        let amp = spike_amplitudes_uv[i].abs();

        let t_bin = ((t / dt).floor() as isize).clamp(0, (num_time_bins - 1) as isize) as usize;
        let z_bin = (((z - depth_min_um) / dz).floor() as isize)
            .clamp(0, (num_depth_bins - 1) as isize) as usize;

        activity_map[t_bin * num_depth_bins + z_bin] += (1.0 + amp).ln();
    }

    // Smooth each depth profile with a [0.25, 0.5, 0.25] 3-tap kernel to stabilize sub-bin registration
    let mut smoothed = activity_map.clone();
    for tb in 0..num_time_bins {
        let off = tb * num_depth_bins;
        for zb in 1..(num_depth_bins - 1) {
            smoothed[off + zb] = 0.25 * activity_map[off + zb - 1]
                + 0.50 * activity_map[off + zb]
                + 0.25 * activity_map[off + zb + 1];
        }
    }

    // Reference depth profile (first active time bin or mean profile)
    let ref_profile = smoothed[0..num_depth_bins].to_vec();
    let max_lag_bins = ((max_drift_um / dz).ceil() as isize).clamp(1, (num_depth_bins as isize) / 2);

    let mut drift_um = vec![0.0f32; num_time_bins];
    let mut time_bin_centers_sec = Vec::with_capacity(num_time_bins);

    for tb in 0..num_time_bins {
        time_bin_centers_sec.push((tb as f64 + 0.5) * dt);
        let cur = &smoothed[tb * num_depth_bins..(tb + 1) * num_depth_bins];

        let mut best_lag = 0isize;
        let mut best_corr = f32::NEG_INFINITY;
        let n_lags = (2 * max_lag_bins + 1) as usize;
        let mut corr_curve = vec![0.0f32; n_lags];

        for (idx, lag) in (-max_lag_bins..=max_lag_bins).enumerate() {
            let mut dot = 0.0f32;
            for zb in 0..num_depth_bins {
                let shifted = zb as isize + lag;
                if shifted >= 0 && (shifted as usize) < num_depth_bins {
                    dot += ref_profile[zb] * cur[shifted as usize];
                }
            }
            corr_curve[idx] = dot;
            if dot > best_corr {
                best_corr = dot;
                best_lag = lag;
            }
        }

        let best_idx = (best_lag + max_lag_bins) as usize;
        let sub_bin = if best_idx > 0 && best_idx + 1 < n_lags {
            // Invert sign so parabolic_subsample_offset (which finds minimum) finds maximum
            parabolic_subsample_offset(
                -corr_curve[best_idx - 1],
                -corr_curve[best_idx],
                -corr_curve[best_idx + 1],
            )
        } else {
            0.0
        };

        drift_um[tb] = (best_lag as f32 + sub_bin) * dz;
    }

    DriftEstimate {
        time_bin_centers_sec,
        drift_um,
        activity_map,
        num_time_bins,
        num_depth_bins,
        depth_min_um,
        depth_bin_size_um: dz,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rigid_drift_estimation_recovers_sinusoidal_motion() {
        let mut times = Vec::new();
        let mut depths = Vec::new();
        let mut amps = Vec::new();

        // Two stationary anatomical layers at y = 100 um and y = 220 um
        // undergoing +15 um drift at t = 1.5s
        let num_bins = 4;
        let true_drifts = [0.0f32, 12.0, -8.0, 4.0];

        for (b, &d_shift) in true_drifts.iter().enumerate() {
            for k in 0..40 {
                let t = b as f64 + (k as f64) * 0.02 + 0.1;
                times.push(t);
                depths.push(100.0 + d_shift);
                amps.push(120.0);

                times.push(t);
                depths.push(220.0 + d_shift);
                amps.push(95.0);
            }
        }

        let est = estimate_rigid_drift(&times, &depths, &amps, 4.0, 1.0, 0.0, 350.0, 4.0, 40.0);
        assert_eq!(est.num_time_bins, num_bins);
        for (b, &expected) in true_drifts.iter().enumerate() {
            assert!(
                (est.drift_um[b] - expected).abs() <= 4.0,
                "Bin {}: est {} vs true {}",
                b,
                est.drift_um[b],
                expected
            );
        }
    }
}
