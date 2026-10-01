use cubecl::prelude::*;
use super::conv::execute_fir_centered;

/// Constructs a normalized 1D symmetric Gaussian FIR kernel of radius $\lceil \text{truncate\_sigma} \cdot \sigma \rceil$:
/// $g[k] = \frac{1}{Z} \exp\left(-\frac{k^2}{2\sigma^2}\right), \quad k \in [-R, R]$.
pub fn gaussian_kernel_1d(sigma_samples: f32, truncate_sigma: f32) -> (Vec<f32>, usize) {
    let sigma = sigma_samples.max(1e-4);
    let radius = (truncate_sigma.max(1.0) * sigma).ceil() as usize;
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

/// Constructs a causal exponential decay kernel $g[k] = \frac{1}{Z}\exp(-k / \tau)$ for $k \in [0, L)$.
pub fn causal_exponential_kernel_1d(tau_samples: f32, truncate_tau: f32) -> Vec<f32> {
    let tau = tau_samples.max(1e-4);
    let len = ((truncate_tau.max(1.0) * tau).ceil() as usize).max(1);
    let mut taps: Vec<f32> = (0..len).map(|k| (-(k as f32) / tau).exp()).collect();
    let sum: f32 = taps.iter().sum();
    if sum > 0.0 {
        for t in &mut taps {
            *t /= sum;
        }
    }
    taps
}

/// Constructs a causal synaptic alpha-function kernel $g[k] = \frac{1}{Z} \frac{k}{\tau} \exp(1 - k / \tau)$ for $k \in [0, L)$.
pub fn causal_alpha_kernel_1d(tau_samples: f32, truncate_tau: f32) -> Vec<f32> {
    let tau = tau_samples.max(1e-4);
    let len = ((truncate_tau.max(1.0) * tau).ceil() as usize).max(2);
    let mut taps: Vec<f32> = (0..len)
        .map(|k| {
            let u = (k as f32) / tau;
            u * (1.0 - u).exp()
        })
        .collect();
    let sum: f32 = taps.iter().sum();
    if sum > 0.0 {
        for t in &mut taps {
            *t /= sum;
        }
    }
    taps
}

/// Applies symmetric zero-phase 1D Gaussian smoothing on the CPU with clamped boundaries.
pub fn gaussian_smooth_1d(signal: &[f32], sigma_samples: f32) -> Vec<f32> {
    if signal.is_empty() || sigma_samples <= 1e-5 {
        return signal.to_vec();
    }
    let (taps, radius) = gaussian_kernel_1d(sigma_samples, 3.0);
    let n = signal.len();
    let r = radius as isize;
    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let mut acc = 0.0f32;
        for (k_idx, &w) in taps.iter().enumerate() {
            let offset = k_idx as isize - r;
            let src = (i as isize + offset).clamp(0, (n - 1) as isize) as usize;
            acc += signal[src] * w;
        }
        out[i] = acc;
    }
    out
}

/// Dispatches multi-channel zero-phase Gaussian smoothing in VRAM via [`execute_fir_centered`].
pub fn execute_gaussian_smooth<R: Runtime>(
    client: &ComputeClient<R>,
    input: &cubecl::server::Handle,
    output: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    sigma_samples: f32,
) {
    let (taps, radius) = gaussian_kernel_1d(sigma_samples, 3.0);
    let taps_handle = client.create_from_slice(f32::as_bytes(&taps));
    execute_fir_centered::<R>(client, input, output, &taps_handle, channels, samples, radius);
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn test_gaussian_smooth_cpu_matches_wgpu() {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let channels = 2;
        let samples = 64;
        let mut data = vec![0.0f32; channels * samples];
        data[32] = 100.0;
        data[samples + 32] = -50.0;

        let in_handle = client.create_from_slice(f32::as_bytes(&data));
        let out_handle = client.empty(channels * samples * 4);

        execute_gaussian_smooth::<WgpuRuntime>(&client, &in_handle, &out_handle, channels, samples, 2.0);
        let gpu_out = f32::from_bytes(&client.read_one_unchecked(out_handle)).to_vec();

        let cpu_ch0 = gaussian_smooth_1d(&data[..samples], 2.0);
        let cpu_ch1 = gaussian_smooth_1d(&data[samples..], 2.0);

        for i in 0..samples {
            assert!((gpu_out[i] - cpu_ch0[i]).abs() < 1e-5);
            assert!((gpu_out[samples + i] - cpu_ch1[i]).abs() < 1e-5);
        }
        // Peak stays centered at index 32
        assert!(gpu_out[32] > gpu_out[31] && gpu_out[32] > gpu_out[33]);
    }
}
