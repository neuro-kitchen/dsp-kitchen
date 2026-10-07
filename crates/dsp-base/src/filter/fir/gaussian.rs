use cubecl::prelude::*;
use super::conv::execute_fir_centered;
use crate::core::{buffer, cast_f32, DspFloat, EdgeMode};

/// Edge handling of Gaussian smoothing, matching `scipy.ndimage.gaussian_filter1d` (`mode="reflect"`).
pub const GAUSSIAN_DEFAULT_EDGE: EdgeMode = EdgeMode::Reflect;

/// Gaussian kernels extend to this many standard deviations (`scipy.ndimage.gaussian_filter1d`'s
/// `truncate` default).
pub const GAUSSIAN_TRUNCATE: f32 = 4.0;

/// Smallest width (σ or τ, in samples) a smoothing kernel is built with; narrower requests are
/// widened to it.
pub const MIN_KERNEL_WIDTH: f32 = 1e-4;

/// Radius in samples of a Gaussian of `sigma_samples` cut at `truncate` σ, rounded as scipy does
/// (`int(truncate · σ + 0.5)`).
pub fn gaussian_radius(sigma_samples: f32, truncate: f32) -> usize {
    (truncate.max(1.0) * sigma_samples.max(MIN_KERNEL_WIDTH) + 0.5) as usize
}

/// Constructs a normalized 1D symmetric Gaussian FIR kernel of radius [`gaussian_radius`]:
/// $g[k] = \frac{1}{Z} \exp\left(-\frac{k^2}{2\sigma^2}\right), \quad k \in [-R, R]$.
pub fn gaussian_kernel_1d(sigma_samples: f32, truncate_sigma: f32) -> (Vec<f32>, usize) {
    let sigma = sigma_samples.max(MIN_KERNEL_WIDTH);
    let radius = gaussian_radius(sigma, truncate_sigma);
    if radius == 0 {
        return (vec![1.0], 0);
    }
    let inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);
    let r = radius as isize;
    let mut taps: Vec<f32> = (-r..=r)
        .map(|k| (-(k as f32).powi(2) * inv_two_sigma2).exp())
        .collect();
    let sum: f32 = taps.iter().sum();
    if sum > 0.0 {
        for t in &mut taps {
            *t /= sum;
        }
    }
    (taps, radius)
}

/// Host reference of zero-phase 1D Gaussian smoothing with reflected edges ([`GAUSSIAN_DEFAULT_EDGE`]),
/// equal to [`execute_gaussian_smooth`] with the default edge.
pub fn gaussian_smooth_1d(signal: &[f32], sigma_samples: f32) -> Vec<f32> {
    if signal.is_empty() || sigma_samples <= MIN_KERNEL_WIDTH {
        return signal.to_vec();
    }
    let (taps, radius) = gaussian_kernel_1d(sigma_samples, GAUSSIAN_TRUNCATE);
    let n = signal.len() as isize;
    let r = radius as isize;
    // scipy "reflect" (d c b a | a b c d | d c b a), one reflection deep, then held at the far end
    let reflect = |p: isize| -> usize {
        let q = if p < 0 { -p - 1 } else if p >= n { 2 * n - 1 - p } else { p };
        q.clamp(0, n - 1) as usize
    };
    (0..n)
        .map(|i| taps.iter().enumerate().map(|(k, &w)| signal[reflect(i + k as isize - r)] * w).sum())
        .collect()
}

/// Multi-channel zero-phase Gaussian smoothing on the device via [`execute_fir_centered`].
pub fn execute_gaussian_smooth<F: DspFloat>(
    client: &Client,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    sigma_samples: f32,
    edge: EdgeMode,
) {
    let (taps, radius) = gaussian_kernel_1d(sigma_samples, GAUSSIAN_TRUNCATE);
    let taps_handle = buffer::upload(client, &cast_f32::<F>(&taps));
    execute_fir_centered::<F>(client, input, output, &taps_handle, channels, samples, radius, edge);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device_matches_host(client: &Client) {
        let channels = 2;
        let samples = 64;
        let mut data = vec![0.0f32; channels * samples];
        data[32] = 100.0;
        data[samples + 32] = -50.0;
        data[1] = 7.0; // near the edge: exercises the reflected border

        let input = buffer::upload(client, &data);
        let output = buffer::empty::<f32>(client, data.len());
        execute_gaussian_smooth::<f32>(client, &input, &output, channels, samples, 2.0, GAUSSIAN_DEFAULT_EDGE);
        let dev = buffer::download::<f32>(client, output);

        for c in 0..channels {
            let host = gaussian_smooth_1d(&data[c * samples..(c + 1) * samples], 2.0);
            for i in 0..samples {
                assert!((dev[c * samples + i] - host[i]).abs() < 1e-5, "{} ch {c} sample {i}", client.name());
            }
        }
        // Peak stays centered at index 32
        assert!(dev[32] > dev[31] && dev[32] > dev[33]);
    }
    runtime_test!(test_gaussian_smooth_device_matches_host, device_matches_host);
}
