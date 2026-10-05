// Parked from crates/dsp-base/src/filter/fir/gaussian.rs (no users).
// Uses `MIN_KERNEL_WIDTH` (filter/fir/gaussian.rs, 1e-4).

/// Constructs a causal exponential decay kernel $g[k] = \frac{1}{Z}\exp(-k / \tau)$ for $k \in [0, L)$.
pub fn causal_exponential_kernel_1d(tau_samples: f32, truncate_tau: f32) -> Vec<f32> {
    let tau = tau_samples.max(MIN_KERNEL_WIDTH);
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
    let tau = tau_samples.max(MIN_KERNEL_WIDTH);
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

