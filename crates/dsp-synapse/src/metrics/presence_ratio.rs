//! Presence Ratio Quality Metric (`presence_ratio.rs`).
//!
//! Measures the fraction of coarse temporal bins across the recording duration in
//! which a unit fires at least `min_spikes_per_bin` spikes (detecting electrode drift
//! or cell death).

/// Computes the Presence Ratio in `[0.0, 1.0]` across `num_bins` coarse temporal divisions.
pub fn compute_presence_ratio(
    spike_samples: &[u64],
    total_recording_samples: u64,
    num_bins: usize,
    min_spikes_per_bin: usize,
) -> f32 {
    let b = num_bins.max(1);
    if spike_samples.is_empty() || total_recording_samples == 0 {
        return 0.0;
    }

    let mut counts = vec![0usize; b];
    let total_f = total_recording_samples as f64;

    for &s in spike_samples {
        let ratio = ((s as f64) / total_f).clamp(0.0, 0.999_999);
        let bin_idx = (ratio * (b as f64)) as usize;
        counts[bin_idx.min(b - 1)] += 1;
    }

    let min_req = min_spikes_per_bin.max(1);
    let active_bins = counts.iter().filter(|&&c| c >= min_req).count();
    (active_bins as f32) / (b as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presence_ratio_full_vs_half_recording() {
        let full: Vec<u64> = (0..100).map(|i| i * 100 + 50).collect();
        assert!((compute_presence_ratio(&full, 10_000, 10, 1) - 1.0).abs() < 1e-6);

        let half: Vec<u64> = (0..50).map(|i| i * 100 + 50).collect();
        assert!((compute_presence_ratio(&half, 10_000, 10, 1) - 0.5).abs() < 1e-6);
    }
}
