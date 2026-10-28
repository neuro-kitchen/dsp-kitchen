use cubecl::prelude::*;
use crate::core::{reduce, DspFloat};

/// `Φ⁻¹(0.75)`: the median absolute value of a zero-mean Gaussian in units of its σ.
pub const MAD_TO_SIGMA: f32 = 0.674_489_75;

/// `2·Φ⁻¹(0.75)`: the interquartile range of a Gaussian in units of its σ.
pub const IQR_TO_SIGMA: f32 = 1.348_979_5;

/// Relative change of the trimmed σ below which [`estimate_noise_trimmed`] stops iterating.
pub const TRIMMED_SIGMA_REL_TOL: f32 = 1e-6;

/// σ below which a signal is treated as constant.
pub const MIN_SIGMA: f32 = 1e-12;

/// Computes the robust estimate of background noise standard deviation (Quiroga et al., 2004):
/// $\sigma_n = \text{median}(|x|) / \Phi^{-1}(0.75)$ ([`MAD_TO_SIGMA`])
pub fn estimate_noise_std(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }

    let mut abs_vals: Vec<f32> = signal.iter().map(|&x| x.abs()).collect();
    let mid = abs_vals.len() / 2;
    abs_vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = abs_vals[mid];

    median / MAD_TO_SIGMA
}

/// [`estimate_noise_std`] of columns `cols` of every channel of the `[channels, samples]` device
/// buffer `input`, on the device (`median(|x|)` at index `n / 2` by
/// [`reduce::row_abs_kth`]); only the `channels` estimates are downloaded. Empty `cols`: zeros.
pub fn execute_channel_noise_std<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    cols: std::ops::Range<usize>,
) -> Vec<f64> {
    if cols.is_empty() || channels == 0 {
        return vec![0.0; channels];
    }
    let out = crate::core::buffer::empty::<F>(client, channels);
    let k = cols.len() / 2;
    reduce::row_abs_kth::<F>(client, input, &out, channels, samples, cols, k);
    crate::core::buffer::download::<F>(client, out)
        .into_iter()
        .map(|m| crate::core::to_f64(m) / MAD_TO_SIGMA as f64)
        .collect()
}

/// Computes the root-mean-square (RMS) standard deviation around the sample mean:
/// $\sigma_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}$
pub fn estimate_noise_rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let n = signal.len() as f32;
    let mean: f32 = signal.iter().sum::<f32>() / n;
    let var: f32 = signal
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    var.sqrt()
}

/// Computes an iterative $k\sigma$-trimmed noise standard deviation for high-firing channels
/// (e.g., dense HD-EMG motor unit action potential trains where MAD can still be biased upward).
pub fn estimate_noise_trimmed(signal: &[f32], clip_sigma: f32, iterations: usize) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let k = clip_sigma.max(1.0);
    let mut sigma = estimate_noise_std(signal);
    if sigma <= MIN_SIGMA {
        return estimate_noise_rms(signal);
    }

    for _ in 0..iterations.max(1) {
        let cutoff = k * sigma;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for &x in signal {
            if x.abs() <= cutoff {
                let xf = x as f64;
                sum += xf;
                sum_sq += xf * xf;
                count += 1;
            }
        }
        if count < 2 {
            break;
        }
        let mean = sum / count as f64;
        let var = (sum_sq / count as f64 - mean * mean).max(0.0);
        // Correct for truncated Gaussian tail variance loss within [-k, k]
        let new_sigma = (var.sqrt()) as f32;
        if (new_sigma - sigma).abs() < TRIMMED_SIGMA_REL_TOL * sigma {
            sigma = new_sigma;
            break;
        }
        sigma = new_sigma;
    }

    sigma
}

/// `max − min` of `signal` (`numpy.ptp`); 0 when empty.
pub fn peak_to_peak(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let (lo, hi) = signal.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    hi - lo
}

/// Computes the Interquartile Range ($\text{IQR} = Q_{75} - Q_{25}$) of a 1D signal (lower ranks
/// `n/4` and `3n/4`, by selection). For a Gaussian, $\sigma \approx \text{IQR}$ / [`IQR_TO_SIGMA`].
pub fn interquartile_range(signal: &[f32]) -> f32 {
    if signal.len() < 2 {
        return 0.0;
    }
    let mut values = signal.to_vec();
    let n = values.len();
    let q75 = *values.select_nth_unstable_by((3 * n) / 4, f32::total_cmp).1;
    let q25 = *values[..(3 * n) / 4].select_nth_unstable_by(n / 4, f32::total_cmp).1;
    q75 - q25
}

/// Computes the Standard Error of the Mean ($\text{SE} = \text{SD} / \sqrt{n}$) from a sample standard
/// deviation and sample count.
#[inline]
pub fn standard_error(sample_std: f32, count: usize) -> f32 {
    if count == 0 {
        0.0
    } else {
        sample_std / (count as f32).sqrt()
    }
}

/// Per-channel mean and population standard deviation of a `[channels, samples]` buffer into
/// `out_mean` / `out_std` (`channels` values each); a parallel reduction per channel
/// ([`reduce::row_mean_std`]).
pub fn execute_channel_mean_std<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    out_mean: &cubecl::server::Handle,
    out_std: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
) {
    reduce::row_mean_std::<F>(client, input, out_mean, out_std, channels, samples);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer;

    #[test]
    fn test_robust_noise_and_trimmed_on_spiky_trace() {
        let n = 10_000;
        let mut trace = Vec::with_capacity(n);
        let mut state = 0x1234_5678u64;
        let mut next_norm = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((state >> 32) as f32 / (u32::MAX as f32)).clamp(1e-6, 1.0 - 1e-6);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (state >> 32) as f32 / (u32::MAX as f32);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        };

        // True background sigma = 5.0 uV, plus 5% large 80 uV spikes
        for i in 0..n {
            let mut v = 5.0 * next_norm();
            if i % 20 == 0 {
                v -= 80.0;
            }
            trace.push(v);
        }

        let mad_std = estimate_noise_std(&trace);
        let trimmed_std = estimate_noise_trimmed(&trace, 3.0, 3);
        let rms_std = estimate_noise_rms(&trace);

        // RMS is heavily inflated by the 80 uV spikes (~18.5 uV)
        assert!(rms_std > 15.0);
        // MAD and trimmed stay close to true 5.0 uV background noise
        assert!((mad_std - 5.0).abs() < 0.8, "mad_std={mad_std}");
        assert!((trimmed_std - 5.0).abs() < 0.6, "trimmed_std={trimmed_std}");
        assert!((standard_error(10.0, 100) - 1.0).abs() < 1e-6);
    }

    fn channel_mean_std(client: &Client) {
        let channels = 4;
        let samples = 256;
        let mut data = vec![0.0f32; channels * samples];
        for c in 0..channels {
            for t in 0..samples {
                let phase = 2.0 * std::f32::consts::PI * (t as f32) / (samples as f32);
                data[c * samples + t] = (c as f32) * 10.0 + (c as f32 + 1.0) * phase.sin();
            }
        }

        let in_handle = buffer::upload(client, &data);
        let mean_handle = buffer::empty::<f32>(client, channels);
        let std_handle = buffer::empty::<f32>(client, channels);

        execute_channel_mean_std::<f32>(
            client,
            &in_handle,
            &mean_handle,
            &std_handle,
            channels,
            samples,
        );

        let means = buffer::download::<f32>(client, mean_handle);
        let stds = buffer::download::<f32>(client, std_handle);

        for c in 0..channels {
            let expected_mean = (c as f32) * 10.0;
            let expected_std = (c as f32 + 1.0) / std::f32::consts::SQRT_2;
            assert!((means[c] - expected_mean).abs() < 1e-3);
            assert!((stds[c] - expected_std).abs() < 1e-2);
        }
    }
    runtime_test!(test_channel_mean_std_kernel, channel_mean_std);
}
