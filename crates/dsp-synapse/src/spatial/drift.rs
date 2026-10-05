//! 2D Depth-Time Activity Histogram & Cross-Correlation Probe Drift Estimation (`drift_map.rs`).
//!
//! Tracks vertical mechanical micromotion $d(t)$ of high-density probes (e.g., Neuropixels)
//! by binning spike depths and log-amplitudes into a 2D spatiotemporal histogram and
//! registering each temporal column via sub-bin parabolic cross-correlation.

use serde::{Deserialize, Serialize};
use dsp_base::math::{cross_correlation, peak_lag};

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

    // Reference depth profile: the first time bin with spikes (drift is relative to it). Bins
    // without spikes cannot be registered: they take the drift of the nearest earlier bin with
    // spikes (or, before the first one, 0).
    let profile = |tb: usize| &smoothed[tb * num_depth_bins..(tb + 1) * num_depth_bins];
    let active: Vec<bool> = (0..num_time_bins).map(|tb| profile(tb).iter().any(|&v| v > 0.0)).collect();
    let time_bin_centers_sec: Vec<f64> = (0..num_time_bins).map(|tb| (tb as f64 + 0.5) * dt).collect();
    let mut drift_um = vec![0.0f32; num_time_bins];
    let Some(ref_bin) = active.iter().position(|&a| a) else {
        return DriftEstimate { time_bin_centers_sec, drift_um, activity_map, num_time_bins, num_depth_bins, depth_min_um, depth_bin_size_um: dz };
    };
    let ref_profile = profile(ref_bin).to_vec();
    let max_lag_bins = ((max_drift_um / dz).ceil() as usize).clamp(1, num_depth_bins / 2);

    for tb in 0..num_time_bins {
        if !active[tb] {
            drift_um[tb] = if tb > 0 { drift_um[tb - 1] } else { 0.0 };
            continue;
        }
        // Shift of this bin's depth profile against the reference, refined between bins
        let corr = cross_correlation(&ref_profile, profile(tb), max_lag_bins);
        let shift_bins = peak_lag(&corr, max_lag_bins).map_or(0.0, |p| p.fractional_lag());
        drift_um[tb] = shift_bins * dz;
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

/// Non-rigid vertical probe drift field $\Delta z(y, t)$ across $B$ depth blocks (`[num_depth_blocks, num_time_bins]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonRigidDriftEstimate {
    pub time_bin_centers_sec: Vec<f64>,
    pub block_centers_um: Vec<f32>,
    /// Row-major `[num_depth_blocks, num_time_bins]` displacement trajectories in $\mu\text{m}$.
    pub block_drift_um: Vec<f32>,
    pub num_depth_blocks: usize,
    pub num_time_bins: usize,
}

impl NonRigidDriftEstimate {
    /// Bilinearly interpolates the non-rigid vertical drift $\Delta z(y, t)$ at depth `y_um` and time `t_sec`.
    pub fn interpolate_drift_at(&self, y_um: f32, t_sec: f64) -> f32 {
        let b = self.num_depth_blocks;
        let t_len = self.num_time_bins;
        if b == 0 || t_len == 0 {
            return 0.0;
        }

        // Helper to interpolate time on block row `b_idx`
        let interp_time = |b_idx: usize| -> f32 {
            let row = &self.block_drift_um[b_idx * t_len..(b_idx + 1) * t_len];
            if t_len == 1 || t_sec <= self.time_bin_centers_sec[0] {
                return row[0];
            }
            if t_sec >= self.time_bin_centers_sec[t_len - 1] {
                return row[t_len - 1];
            }
            let pos = self
                .time_bin_centers_sec
                .partition_point(|&tc| tc < t_sec)
                .clamp(1, t_len - 1);
            let t0 = self.time_bin_centers_sec[pos - 1];
            let t1 = self.time_bin_centers_sec[pos];
            let alpha = ((t_sec - t0) / (t1 - t0).max(1e-9)) as f32;
            row[pos - 1] * (1.0 - alpha) + row[pos] * alpha
        };

        if b == 1 || y_um <= self.block_centers_um[0] {
            return interp_time(0);
        }
        if y_um >= self.block_centers_um[b - 1] {
            return interp_time(b - 1);
        }

        let b_pos = self
            .block_centers_um
            .partition_point(|&yc| yc < y_um)
            .clamp(1, b - 1);
        let y0 = self.block_centers_um[b_pos - 1];
        let y1 = self.block_centers_um[b_pos];
        let beta = (y_um - y0) / (y1 - y0).max(1e-6);
        interp_time(b_pos - 1) * (1.0 - beta) + interp_time(b_pos) * beta
    }
}

/// Estimates non-rigid multi-depth vertical probe drift $\Delta z(y, t)$ by partitioning
/// `[depth_min_um, depth_max_um]` into `num_depth_blocks` overlapping windows and registering each block.
#[allow(clippy::too_many_arguments)]
pub fn estimate_nonrigid_drift(
    spike_times_sec: &[f64],
    spike_depths_um: &[f32],
    spike_amplitudes_uv: &[f32],
    total_duration_sec: f64,
    time_bin_sec: f64,
    depth_min_um: f32,
    depth_max_um: f32,
    depth_bin_size_um: f32,
    max_drift_um: f32,
    num_depth_blocks: usize,
) -> NonRigidDriftEstimate {
    let b_count = num_depth_blocks.max(1);
    let span = (depth_max_um - depth_min_um).max(depth_bin_size_um * 4.0);
    let block_step = span / (b_count as f32);
    // Use 25% overlap on each side so boundary transitions are smooth
    let half_window = (block_step * 0.65).max(depth_bin_size_um * 2.0);

    let mut block_centers_um = Vec::with_capacity(b_count);
    let mut block_drift_um = Vec::new();
    let mut time_bin_centers_sec = Vec::new();
    let mut num_time_bins = 0;

    let n = spike_times_sec
        .len()
        .min(spike_depths_um.len())
        .min(spike_amplitudes_uv.len());

    for b in 0..b_count {
        let center = depth_min_um + (b as f32 + 0.5) * block_step;
        let lo = (center - half_window).max(depth_min_um);
        let hi = (center + half_window).min(depth_max_um);
        block_centers_um.push(center);

        let mut b_times = Vec::new();
        let mut b_depths = Vec::new();
        let mut b_amps = Vec::new();
        for i in 0..n {
            let z = spike_depths_um[i];
            if z >= lo && z <= hi {
                b_times.push(spike_times_sec[i]);
                b_depths.push(z);
                b_amps.push(spike_amplitudes_uv[i]);
            }
        }

        let est = estimate_rigid_drift(
            &b_times,
            &b_depths,
            &b_amps,
            total_duration_sec,
            time_bin_sec,
            lo,
            hi,
            depth_bin_size_um,
            max_drift_um,
        );
        if b == 0 {
            num_time_bins = est.num_time_bins;
            time_bin_centers_sec = est.time_bin_centers_sec;
        }
        block_drift_um.extend_from_slice(&est.drift_um);
    }

    NonRigidDriftEstimate {
        time_bin_centers_sec,
        block_centers_um,
        block_drift_um,
        num_depth_blocks: b_count,
        num_time_bins,
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

    #[test]
    fn test_rigid_drift_with_empty_bins_uses_first_active_bin() {
        // Bins 0 and 2 have no spikes; bin 1 is the reference, bin 3 is shifted by +12 µm
        let (mut times, mut depths, mut amps) = (Vec::new(), Vec::new(), Vec::new());
        for (b, shift) in [(1usize, 0.0f32), (3, 12.0)] {
            for k in 0..40 {
                let t = b as f64 + k as f64 * 0.02 + 0.1;
                for layer in [100.0f32, 220.0] {
                    times.push(t);
                    depths.push(layer + shift);
                    amps.push(100.0);
                }
            }
        }
        let est = estimate_rigid_drift(&times, &depths, &amps, 4.0, 1.0, 0.0, 350.0, 4.0, 40.0);
        assert_eq!(est.drift_um[0], 0.0, "before the reference");
        assert!(est.drift_um[1].abs() <= 1e-3, "reference bin: {}", est.drift_um[1]);
        assert_eq!(est.drift_um[2], est.drift_um[1], "empty bin holds the previous drift");
        assert!((est.drift_um[3] - 12.0).abs() <= 4.0, "shifted bin: {}", est.drift_um[3]);
    }

    #[test]
    fn test_nonrigid_drift_recovers_depth_dependent_gradient() {
        let mut times = Vec::new();
        let mut depths = Vec::new();
        let mut amps = Vec::new();

        // Bottom block (y ~ 100 um) drifts by +12 um at t=1.5s; top block (y ~ 500 um) drifts by -12 um
        for b in 0..2 {
            let bottom_shift = if b == 0 { 0.0 } else { 12.0 };
            let top_shift = if b == 0 { 0.0 } else { -12.0 };
            for k in 0..40 {
                let t = b as f64 + (k as f64) * 0.02 + 0.1;
                times.push(t);
                depths.push(100.0 + bottom_shift);
                amps.push(110.0);

                times.push(t);
                depths.push(500.0 + top_shift);
                amps.push(110.0);
            }
        }

        let nr = estimate_nonrigid_drift(&times, &depths, &amps, 2.0, 1.0, 0.0, 600.0, 4.0, 30.0, 2);
        let d_bottom = nr.interpolate_drift_at(100.0, 1.5);
        let d_top = nr.interpolate_drift_at(500.0, 1.5);
        assert!((d_bottom - 12.0).abs() <= 4.0, "d_bottom={d_bottom}");
        assert!((d_top - (-12.0)).abs() <= 4.0, "d_top={d_top}");
    }
}
