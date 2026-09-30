//! Amplitude cutoff (SpikeInterface `amplitude_cutoff`, 0.105): estimated fraction of spikes
//! missed below the detection threshold, from the left tail of the amplitude histogram.

/// Default number of histogram bins.
pub const AMPLITUDE_CUTOFF_BINS: usize = 100;
/// Default Gaussian smoothing (in bins) of the histogram.
pub const AMPLITUDE_CUTOFF_SMOOTHING: f64 = 3.0;
/// Minimum spikes per bin; below it the metric is NaN.
pub const AMPLITUDE_CUTOFF_MIN_RATIO: f64 = 5.0;

/// Amplitude cutoff in `[0, 0.5]` (NaN with fewer than 5 spikes per bin) with SpikeInterface's
/// defaults. Amplitudes are magnitudes: negative troughs are used by absolute value (SpikeInterface
/// inverts them for `peak_sign="neg"`).
pub fn compute_amplitude_cutoff<T: Into<f64> + Copy>(amplitudes: &[T]) -> f64 {
    compute_amplitude_cutoff_with(amplitudes, AMPLITUDE_CUTOFF_BINS, AMPLITUDE_CUTOFF_SMOOTHING, AMPLITUDE_CUTOFF_MIN_RATIO)
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

    // Missing spikes: smoothed counts after the last bin at least as high as the first one.
    let cutoff_point = pdf[0];
    let g = pdf.iter().rposition(|&v| v >= cutoff_point).unwrap_or(0);
    let missed: i64 = pdf[g + 1..].iter().sum();
    let fraction = missed as f64 / (n as f64 + missed as f64);
    fraction.min(0.5)
}

/// `numpy.histogram(values, bins)` counts with equal-width bins over `[min, max]`.
fn histogram(values: &[f64], bins: usize) -> Vec<i64> {
    let (mut lo, mut hi) = values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
    if lo == hi {
        lo -= 0.5;
        hi += 0.5;
    }
    let edges: Vec<f64> = (0..=bins).map(|i| lo + (hi - lo) * i as f64 / bins as f64).collect();
    let norm = bins as f64 / (hi - lo);
    let mut counts = vec![0i64; bins];
    for &v in values {
        let mut i = ((v - lo) * norm) as usize;
        if i >= bins {
            i = bins - 1;
        }
        // numpy's corrections for rounding at the edges
        if v < edges[i] && i > 0 {
            i -= 1;
        } else if i + 1 < bins && v >= edges[i + 1] {
            i += 1;
        }
        counts[i] += 1;
    }
    counts
}

/// `scipy.ndimage.gaussian_filter1d(counts, sigma, mode="nearest")` on an integer array: the output
/// keeps the integer type, so values are truncated.
fn gaussian_filter1d_nearest_int(counts: &[i64], sigma: f64) -> Vec<i64> {
    let radius = (4.0 * sigma + 0.5) as i64;
    let sigma2 = sigma * sigma;
    let mut weights: Vec<f64> = (-radius..=radius).map(|x| (-0.5 / sigma2 * (x * x) as f64).exp()).collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_smoothing_matches_scipy() {
        // scipy.ndimage.gaussian_filter1d(np.array([...], np.int64), 3, mode="nearest")
        let h = [0, 1, 2, 3, 10, 0, 0, 7, 1, 1, 0, 0, 0, 5, 5, 5, 9, 0, 0, 1];
        let expected = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 1];
        assert_eq!(gaussian_filter1d_nearest_int(&h, 3.0), expected);
    }

    #[test]
    fn too_few_spikes_is_nan() {
        assert!(compute_amplitude_cutoff(&[1.0f32; 499]).is_nan());
    }
}
