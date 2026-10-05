//! Firing-train quality metrics: Inter-Spike Interval (ISI) violations, presence ratio,
//! amplitude cutoff, and refractory-period contamination (Allen / IBL / SpikeInterface standard).

/// Default number of histogram bins for amplitude cutoff.
pub const AMPLITUDE_CUTOFF_BINS: usize = 100;
/// Default Gaussian smoothing (in bins) of the amplitude histogram.
pub const AMPLITUDE_CUTOFF_SMOOTHING: f64 = 3.0;
/// Minimum spikes per bin; below it the metric is NaN.
pub const AMPLITUDE_CUTOFF_MIN_RATIO: f64 = 5.0;

/// ISI-violation metrics of one unit.
#[derive(Debug, Clone, PartialEq)]
pub struct IsiMetrics {
    pub total_spikes: usize,
    /// Consecutive intervals shorter than the threshold.
    pub violation_count: usize,
    /// `violation_count` as a percentage of the intervals.
    pub violation_rate_pct: f32,
    /// Hill et al. (2011) ratio of the violation rate to the unit's rate (`isi_violations_ratio`); NaN without spikes.
    pub isi_violations_ratio: f64,
    /// Violations per second of recording (`isi_violations_rate`).
    pub violations_per_sec: f64,
    /// Spikes per second of recording.
    pub firing_rate_hz: f64,
}

/// ISI violations of a spike train (sample indices, any order) over a recording of `total_duration_sec`.
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
        violation_rate_pct: if n > 1 {
            violations as f32 / (n - 1) as f32 * 100.0
        } else {
            0.0
        },
        isi_violations_ratio: ratio,
        violations_per_sec: per_sec,
        firing_rate_hz: if total_duration_sec > 0.0 {
            n as f64 / total_duration_sec
        } else {
            0.0
        },
    }
}

/// Inter-spike-interval histogram of a spike train (sample indices, any order): counts of the
/// intervals in `bin_ms` bins over `[0, max_ms)`, and the bin centres in ms.
pub fn isi_histogram(spike_samples: &[u64], sample_rate_hz: f64, bin_ms: f64, max_ms: f64) -> (Vec<f64>, Vec<u64>) {
    let bin_ms = bin_ms.max(1e-3);
    let bins = ((max_ms.max(bin_ms) / bin_ms).round() as usize).max(1);
    let hi = bins as f64 * bin_ms;
    let mut sorted = spike_samples.to_vec();
    sorted.sort_unstable();
    let ms = 1e3 / sample_rate_hz.max(f64::MIN_POSITIVE);
    // Intervals of exactly `hi` belong to the next bin, not the last one
    let intervals = sorted.windows(2).map(|w| (w[1] - w[0]) as f64 * ms).filter(|&d| d < hi);
    (dsp_base::math::bin_centers(0.0, hi, bins), dsp_base::math::histogram(intervals, 0.0, hi, bins))
}

/// Fraction of `bin_duration_s` bins in which the unit fires more than `mean_fr_ratio_thresh` × its mean rate.
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

/// Amplitude cutoff in `[0, 0.5]` (NaN with fewer than 5 spikes per bin) with SpikeInterface's defaults.
pub fn compute_amplitude_cutoff<T: Into<f64> + Copy>(amplitudes: &[T]) -> f64 {
    compute_amplitude_cutoff_with(
        amplitudes,
        AMPLITUDE_CUTOFF_BINS,
        AMPLITUDE_CUTOFF_SMOOTHING,
        AMPLITUDE_CUTOFF_MIN_RATIO,
    )
}

/// [`compute_amplitude_cutoff`] with explicit histogram parameters.
pub fn compute_amplitude_cutoff_with<T: Into<f64> + Copy>(
    amplitudes: &[T],
    num_bins: usize,
    smoothing_bins: f64,
    min_ratio: f64,
) -> f64 {
    let amps: Vec<f64> = amplitudes.iter().map(|&a| a.into().abs()).collect();
    let n = amps.len();
    if num_bins == 0 || (n as f64) / (num_bins as f64) < min_ratio {
        return f64::NAN;
    }
    let counts = histogram(&amps, num_bins);
    let pdf = gaussian_filter1d_nearest_int(&counts, smoothing_bins);

    let cutoff_point = pdf[0];
    let g = pdf.iter().rposition(|&v| v >= cutoff_point).unwrap_or(0);
    let missed: i64 = pdf[g + 1..].iter().sum();
    let fraction = missed as f64 / (n as f64 + missed as f64);
    fraction.min(0.5)
}

fn histogram(values: &[f64], bins: usize) -> Vec<i64> {
    let (mut lo, mut hi) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
    if lo == hi {
        lo -= 0.5;
        hi += 0.5;
    }
    let edges: Vec<f64> = (0..=bins)
        .map(|i| lo + (hi - lo) * i as f64 / bins as f64)
        .collect();
    let norm = bins as f64 / (hi - lo);
    let mut counts = vec![0i64; bins];
    for &v in values {
        let mut i = ((v - lo) * norm) as usize;
        if i >= bins {
            i = bins - 1;
        }
        if v < edges[i] && i > 0 {
            i -= 1;
        } else if i + 1 < bins && v >= edges[i + 1] {
            i += 1;
        }
        counts[i] += 1;
    }
    counts
}

fn gaussian_filter1d_nearest_int(counts: &[i64], sigma: f64) -> Vec<i64> {
    let radius = (4.0 * sigma + 0.5) as i64;
    let sigma2 = sigma * sigma;
    let mut weights: Vec<f64> = (-radius..=radius)
        .map(|x| (-0.5 / sigma2 * (x * x) as f64).exp())
        .collect();
    let sum: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= sum);
    let last = counts.len() as i64 - 1;
    let at = |i: i64| counts[i.clamp(0, last) as usize] as f64;
    let center = radius as usize;
    (0..counts.len() as i64)
        .map(|i| {
            let mut acc = at(i) * weights[center];
            for j in 1..=radius {
                acc += (at(i + j) + at(i - j)) * weights[center + j as usize];
            }
            acc as i64
        })
        .collect()
}

fn ms_to_samples(ms: f64, sample_rate_hz: f64) -> u64 {
    (ms * sample_rate_hz * 1e-3).round_ties_even().max(0.0) as u64
}

/// Number of spike pairs (not only consecutive ones) closer than or equal to `t_r` samples in a sorted train.
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

/// Refractory-period contamination of a unit (Llobet et al. 2022).
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
    let denom =
        1.0 - n_v * (total_samples as f64 - 2.0 * n * t_c) / (n * n * (t_r as f64 - t_c));
    if denom < 0.0 {
        1.0
    } else {
        1.0 - denom.sqrt()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_isi_histogram() {
        // 1 kHz: intervals of 1, 2, 2 and 10 ms (10 ms is outside [0, 10))
        let (centers, counts) = isi_histogram(&[0, 1, 3, 5, 15], 1000.0, 1.0, 10.0);
        assert_eq!(centers.len(), 10);
        assert_eq!(centers[0], 0.5);
        assert_eq!(counts[1], 1);
        assert_eq!(counts[2], 2);
        assert_eq!(counts.iter().sum::<u64>(), 3);
    }

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

    #[test]
    fn test_presence_ratio_full_vs_half_recording() {
        let full: Vec<u64> = (0..100).map(|i| i * 100 + 50).collect();
        assert!((compute_presence_ratio(&full, 10_000, 1_000.0, 1.0, 0.0) - 1.0).abs() < 1e-12);
        let half: Vec<u64> = (0..50).map(|i| i * 100 + 50).collect();
        assert!((compute_presence_ratio(&half, 10_000, 1_000.0, 1.0, 0.0) - 0.5).abs() < 1e-12);
        assert!(compute_presence_ratio(&half, 500, 1_000.0, 1.0, 0.0).is_nan());
    }

    #[test]
    fn integer_smoothing_matches_scipy() {
        let h = [0, 1, 2, 3, 10, 0, 0, 7, 1, 1, 0, 0, 0, 5, 5, 5, 9, 0, 0, 1];
        let expected = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 1];
        assert_eq!(gaussian_filter1d_nearest_int(&h, 3.0), expected);
    }

    #[test]
    fn too_few_spikes_is_nan() {
        assert!(compute_amplitude_cutoff(&[1.0f32; 499]).is_nan());
    }

    #[test]
    fn violations_count_all_close_pairs() {
        assert_eq!(count_refractory_violations(&[0, 10, 20, 100], 25), 3);
        assert!(compute_llobet_contamination(&[5], 1000, 30_000.0, 1.0, 0.0).is_nan());
        assert_eq!(
            compute_llobet_contamination(&[0, 1_000, 2_000], 30_000, 30_000.0, 1.0, 0.0),
            0.0
        );
    }
}
