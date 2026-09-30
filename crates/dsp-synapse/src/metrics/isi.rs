//! Inter-spike-interval violations (SpikeInterface `isi_violations`, Hill et al. 2011).

/// ISI-violation metrics of one unit.
#[derive(Debug, Clone, PartialEq)]
pub struct IsiMetrics {
    pub total_spikes: usize,
    /// Consecutive intervals shorter than the threshold.
    pub violation_count: usize,
    /// `violation_count` as a percentage of the intervals.
    pub violation_rate_pct: f32,
    /// Hill et al. (2011) ratio of the violation rate to the unit's rate (SpikeInterface
    /// `isi_violations_ratio`); NaN without spikes.
    pub isi_violations_ratio: f64,
    /// Violations per second of recording (SpikeInterface `isi_violations_rate`).
    pub violations_per_sec: f64,
    /// Spikes per second of recording.
    pub firing_rate_hz: f64,
}

/// ISI violations of a spike train (sample indices, any order) over a recording of
/// `total_duration_sec`. An interval is a violation when shorter than `isi_threshold_ms`;
/// `min_isi_ms` is the censored period imposed by acquisition / sorting (SpikeInterface defaults:
/// 1.5 ms and 0 ms). Matches `spikeinterface.metrics.quality.misc_metrics.isi_violations`.
pub fn compute_isi_violations(
    spike_samples: &[u64],
    sample_rate_hz: f64,
    total_duration_sec: f64,
    isi_threshold_ms: f64,
    min_isi_ms: f64,
) -> IsiMetrics {
    let mut sorted = spike_samples.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let threshold_s = isi_threshold_ms * 1e-3;
    // Seconds per spike first, then differences, as SpikeInterface does (same rounding at the threshold).
    let violations = sorted
        .windows(2)
        .filter(|w| w[1] as f64 / sample_rate_hz - w[0] as f64 / sample_rate_hz < threshold_s)
        .count();

    let (ratio, per_sec) = if n > 0 {
        let violation_time = 2.0 * n as f64 * (threshold_s - min_isi_ms * 1e-3);
        let total_rate = n as f64 / total_duration_sec;
        let violation_rate = violations as f64 / violation_time;
        (violation_rate / total_rate, violations as f64 / total_duration_sec)
    } else {
        (f64::NAN, f64::NAN)
    };

    IsiMetrics {
        total_spikes: n,
        violation_count: violations,
        violation_rate_pct: if n > 1 { violations as f32 / (n - 1) as f32 * 100.0 } else { 0.0 },
        isi_violations_ratio: ratio,
        violations_per_sec: per_sec,
        firing_rate_hz: if total_duration_sec > 0.0 { n as f64 / total_duration_sec } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isi_violations() {
        let spikes = vec![300, 0, 30];
        let metrics = compute_isi_violations(&spikes, 30000.0, 1.0, 1.5, 0.0);
        assert_eq!(metrics.total_spikes, 3);
        assert_eq!(metrics.violation_count, 1);
        assert_eq!(metrics.violation_rate_pct, 50.0);
        assert_eq!(metrics.firing_rate_hz, 3.0);
    }
}
