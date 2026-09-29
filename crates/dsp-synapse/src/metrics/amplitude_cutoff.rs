//! Amplitude Cutoff Quality Metric (`amplitude_cutoff.rs`).
//!
//! Estimates the fraction of false-negative spikes missed below the detection
//! threshold by fitting a Gaussian distribution to the upper half of the amplitude
//! histogram and integrating its truncated tail (matching `spikeinterface.qualitymetrics.compute_amplitude_cutoffs`).

/// Approximates the standard normal cumulative distribution function $\Phi(z)$.
fn normal_cdf(z: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 rational approximation
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = sign * (1.0 - poly * (-x * x).exp());
    0.5 * (1.0 + erf)
}

/// Computes the Allen / IBL Amplitude Cutoff metric in `[0.0, 0.5]`.
///
/// A well-isolated single unit whose amplitude distribution is well above the detection
/// threshold has `amplitude_cutoff < 0.01`.
pub fn compute_amplitude_cutoff(amplitudes_uv: &[f32]) -> f32 {
    let n = amplitudes_uv.len();
    if n < 10 {
        return 0.5;
    }

    let mut abs_amps: Vec<f64> = amplitudes_uv
        .iter()
        .map(|&a| a.abs() as f64)
        .filter(|v| v.is_finite())
        .collect();
    if abs_amps.len() < 10 {
        return 0.5;
    }
    abs_amps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_amp = abs_amps[0];
    let max_amp = abs_amps[abs_amps.len() - 1];
    let span = max_amp - min_amp;
    if span <= 1e-9 {
        return 0.5;
    }

    // Build a histogram of |amplitude| to locate the distribution mode (peak bin).
    // Unlike the sample median, the histogram mode remains anchored near the true mean
    // even when the lower tail is truncated by the spike detection threshold.
    let num_bins = (abs_amps.len() / 12).clamp(10, 40);
    let bin_width = span / (num_bins as f64);
    let mut counts = vec![0.0f64; num_bins];
    for &v in &abs_amps {
        let idx = (((v - min_amp) / bin_width).floor() as usize).min(num_bins - 1);
        counts[idx] += 1.0;
    }

    // 3-point binomial smoothing [0.25, 0.5, 0.25] for robust mode identification
    let mut best_bin = 0usize;
    let mut best_density = -1.0f64;
    for b in 0..num_bins {
        let left = if b > 0 { counts[b - 1] } else { counts[b] };
        let right = if b + 1 < num_bins { counts[b + 1] } else { counts[b] };
        let smooth = 0.25 * left + 0.5 * counts[b] + 0.25 * right;
        if smooth > best_density {
            best_density = smooth;
            best_bin = b;
        }
    }

    let mode = min_amp + (best_bin as f64 + 0.5) * bin_width;

    // Estimate standard deviation from the untruncated upper tail (v >= mode)
    let mut sum_sq = 0.0f64;
    let mut count_upper = 0usize;
    for &v in &abs_amps {
        if v >= mode {
            let d = v - mode;
            sum_sq += d * d;
            count_upper += 1;
        }
    }
    let sigma = (sum_sq / (count_upper.max(1) as f64)).sqrt().max(1e-6);

    // Tail probability below min_amp under N(mode, sigma^2)
    let z = (min_amp - mode) / sigma;
    (normal_cdf(z) as f32).clamp(0.0, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amplitude_cutoff_clean_vs_truncated() {
        // Clean bell-shaped unit well above threshold (mode = 120 uV, min ~ 95 uV)
        let mut clean = Vec::new();
        for i in 0..200 {
            let u = (i as f32) / 100.0 - 1.0; // [-1, 1]
            let z = u * (0.15 + 0.85 * u * u * u * u) * 2.5;
            clean.push(-120.0 + z * 10.0);
        }
        let cutoff_clean = compute_amplitude_cutoff(&clean);
        assert!(cutoff_clean < 0.01, "cutoff_clean = {}", cutoff_clean);

        // Heavily truncated unit: only the upper half [100 uV, 130 uV] of a bell curve
        // centered at 100 uV survived the 100 uV detection threshold
        let mut truncated = Vec::new();
        for i in 0..200 {
            let u = (i as f32) / 200.0; // [0, 1] upper half only!
            let z = u * (0.2 + 0.8 * u * u) * 3.0;
            truncated.push(-(100.0 + z * 10.0));
        }
        let cutoff_trunc = compute_amplitude_cutoff(&truncated);
        assert!(cutoff_trunc > 0.35, "cutoff_trunc = {}", cutoff_trunc);
    }
}
