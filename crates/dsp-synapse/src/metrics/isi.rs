/// Result of Inter-Spike Interval (ISI) violation analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct IsiMetrics {
    pub total_spikes: usize,
    pub violation_count: usize,
    /// Percentage of intervals that violate the biological refractory period (typically < 1.5 ms).
    pub violation_rate_pct: f32,
    /// Mean firing rate in Hz.
    pub firing_rate_hz: f32,
}

/// Evaluates refractory period violations for a spike train.
///
/// A single biological neuron cannot fire two action potentials within its refractory period
/// (typically 1.5 ms). High violation rates indicate multi-unit contamination or noise.
pub fn compute_isi_violations(
    spike_samples: &[u64],
    sample_rate_hz: f64,
    refractory_ms: f64,
) -> IsiMetrics {
    if spike_samples.len() < 2 {
        return IsiMetrics {
            total_spikes: spike_samples.len(),
            violation_count: 0,
            violation_rate_pct: 0.0,
            firing_rate_hz: 0.0,
        };
    }

    let refractory_samples = (refractory_ms * 0.001 * sample_rate_hz).round() as u64;
    let mut violations = 0usize;

    for i in 1..spike_samples.len() {
        let diff = spike_samples[i].saturating_sub(spike_samples[i - 1]);
        if diff < refractory_samples {
            violations += 1;
        }
    }

    let total_intervals = (spike_samples.len() - 1) as f32;
    let rate_pct = (violations as f32 / total_intervals) * 100.0;

    let duration_sec = (spike_samples.last().unwrap() - spike_samples.first().unwrap()) as f32
        / sample_rate_hz as f32;
    let firing_rate = if duration_sec > 0.0 {
        spike_samples.len() as f32 / duration_sec
    } else {
        0.0
    };

    IsiMetrics {
        total_spikes: spike_samples.len(),
        violation_count: violations,
        violation_rate_pct: rate_pct,
        firing_rate_hz: firing_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isi_violations() {
        // Spikes at 0, 30, 300 samples @ 30kHz (30 samples = 1.0 ms < 1.5 ms -> violation!)
        let spikes = vec![0, 30, 300];
        let metrics = compute_isi_violations(&spikes, 30000.0, 1.5);
        assert_eq!(metrics.total_spikes, 3);
        assert_eq!(metrics.violation_count, 1);
        assert_eq!(metrics.violation_rate_pct, 50.0);
    }
}
