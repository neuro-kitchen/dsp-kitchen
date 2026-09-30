//! Refractory-period contamination (Llobet et al. 2022, SpikeInterface `rp_contamination`).
//!
//! The Hill et al. (2011) estimate is [`super::IsiMetrics::isi_violations_ratio`].

/// Python-style `int(round(x))` (ties to even), as SpikeInterface converts ms to samples.
fn ms_to_samples(ms: f64, sample_rate_hz: f64) -> u64 {
    (ms * sample_rate_hz * 1e-3).round_ties_even().max(0.0) as u64
}

/// Number of spike pairs (not only consecutive ones) closer than or equal to `t_r` samples in a
/// sorted train (SpikeInterface `_compute_nb_violations_numba`).
pub fn count_refractory_violations(sorted_samples: &[u64], t_r: u64) -> u64 {
    let mut n_v = 0u64;
    for (i, &a) in sorted_samples.iter().enumerate() {
        for &b in &sorted_samples[i + 1..] {
            if b - a > t_r {
                break;
            }
            n_v += 1;
        }
    }
    n_v
}

/// Refractory-period contamination of a unit (spike sample indices, any order) in a recording of
/// `total_samples` (Llobet et al. 2022; matches SpikeInterface `compute_refrac_period_violations`,
/// defaults 1 ms refractory and 0 ms censored). NaN for fewer than two spikes.
pub fn compute_llobet_contamination(
    spike_samples: &[u64],
    total_samples: u64,
    sample_rate_hz: f64,
    refractory_period_ms: f64,
    censored_period_ms: f64,
) -> f64 {
    let n = spike_samples.len();
    if n <= 1 {
        return f64::NAN;
    }
    let mut sorted = spike_samples.to_vec();
    sorted.sort_unstable();
    let t_c = ms_to_samples(censored_period_ms, sample_rate_hz) as f64;
    let t_r = ms_to_samples(refractory_period_ms, sample_rate_hz);
    let n_v = count_refractory_violations(&sorted, t_r) as f64;
    let n = n as f64;
    let denom = 1.0 - n_v * (total_samples as f64 - 2.0 * n * t_c) / (n * n * (t_r as f64 - t_c));
    if denom < 0.0 { 1.0 } else { 1.0 - denom.sqrt() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn violations_count_all_close_pairs() {
        // 0-10-20 are all within 25 samples of each other: 3 pairs; 100 is alone.
        assert_eq!(count_refractory_violations(&[0, 10, 20, 100], 25), 3);
        assert_eq!(compute_llobet_contamination(&[5], 1000, 30_000.0, 1.0, 0.0).is_nan(), true);
        assert_eq!(compute_llobet_contamination(&[0, 1_000, 2_000], 30_000, 30_000.0, 1.0, 0.0), 0.0);
    }
}
