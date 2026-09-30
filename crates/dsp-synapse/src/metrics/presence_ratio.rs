//! Presence ratio (SpikeInterface `presence_ratio`): fraction of fixed-length bins in which the
//! unit fires.

/// Fraction of `bin_duration_s` bins (starting at sample 0; a trailing partial bin is dropped, as in
/// SpikeInterface) in which the unit fires more than `mean_fr_ratio_thresh` × its mean rate
/// (default 0: at least one spike). NaN without spikes or when the recording is shorter than a bin.
pub fn compute_presence_ratio(
    spike_samples: &[u64],
    total_samples: u64,
    sample_rate_hz: f64,
    bin_duration_s: f64,
    mean_fr_ratio_thresh: f64,
) -> f64 {
    let bin = (bin_duration_s * sample_rate_hz) as u64;
    if spike_samples.is_empty() || bin == 0 || total_samples < bin {
        return f64::NAN;
    }
    let num_bins = (total_samples / bin) as usize;
    let last_edge = num_bins as u64 * bin;
    let mut counts = vec![0u64; num_bins];
    for &s in spike_samples {
        if s <= last_edge {
            counts[((s / bin) as usize).min(num_bins - 1)] += 1;
        }
    }
    let unit_fr = spike_samples.len() as f64 / (total_samples as f64 / sample_rate_hz);
    let threshold = (unit_fr * bin_duration_s * mean_fr_ratio_thresh).floor() as u64;
    counts.iter().filter(|&&c| c > threshold).count() as f64 / num_bins as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presence_ratio_full_vs_half_recording() {
        let full: Vec<u64> = (0..100).map(|i| i * 100 + 50).collect();
        assert!((compute_presence_ratio(&full, 10_000, 1_000.0, 1.0, 0.0) - 1.0).abs() < 1e-12);
        let half: Vec<u64> = (0..50).map(|i| i * 100 + 50).collect();
        assert!((compute_presence_ratio(&half, 10_000, 1_000.0, 1.0, 0.0) - 0.5).abs() < 1e-12);
        assert!(compute_presence_ratio(&half, 500, 1_000.0, 1.0, 0.0).is_nan());
    }
}
