use std::f32::consts::PI;

/// `|x|` below which [`sinc`] returns its limit 1.
const SINC_ZERO: f32 = 1e-7;

/// 4-term Blackman-Harris coefficients `a0..a3`.
const BLACKMAN_HARRIS: [f32; 4] = [0.35875, 0.48829, 0.14128, 0.01168];

/// Hamming window `α`, `β` (`α − β·cos`).
const HAMMING: [f32; 2] = [0.54, 0.46];

/// Smallest half-width a Blackman-Harris window is evaluated over.
const MIN_HALF_WIDTH: f32 = 1e-12;

/// Smallest σ (samples) a Gaussian window is built with.
const MIN_WINDOW_SIGMA: f32 = 1e-6;

/// Computes normalized sinc: $\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$, with $\text{sinc}(0) = 1$.
#[inline]
pub fn sinc(x: f32) -> f32 {
    if x.abs() < SINC_ZERO {
        1.0
    } else {
        let pix = PI * x;
        pix.sin() / pix
    }
}

/// Blackman-Harris 4-term apodization window evaluated at `x` over `[-half_width, half_width]`.
#[inline]
pub fn blackman_window(x: f32, half_width: f32) -> f32 {
    let norm = (x / half_width.max(MIN_HALF_WIDTH)).clamp(-1.0, 1.0);
    let u = (norm + 1.0) * 0.5;
    let [a0, a1, a2, a3] = BLACKMAN_HARRIS;

    let two_pi_u = 2.0 * PI * u;
    a0 - a1 * two_pi_u.cos() + a2 * (2.0 * two_pi_u).cos() - a3 * (3.0 * two_pi_u).cos()
}

/// Generates a symmetric 1D Gaussian window of length `len` with standard deviation `sigma` (in samples).
pub fn gaussian_window(len: usize, sigma: f32) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![1.0];
    }
    let center = (len - 1) as f32 * 0.5;
    let inv_two_sigma2 = 1.0 / (2.0 * sigma.max(MIN_WINDOW_SIGMA).powi(2));
    (0..len)
        .map(|i| {
            let d = i as f32 - center;
            (-d * d * inv_two_sigma2).exp()
        })
        .collect()
}

/// Generates a symmetric Hann window of length `len`.
pub fn hann_window(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![1.0];
    }
    let denom = (len - 1) as f32;
    (0..len)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f32) / denom).cos()))
        .collect()
}

/// Generates a symmetric Hamming window of length `len`.
pub fn hamming_window(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![1.0];
    }
    let denom = (len - 1) as f32;
    (0..len)
        .map(|i| HAMMING[0] - HAMMING[1] * (2.0 * PI * (i as f32) / denom).cos())
        .collect()
}

/// Relative size of the last series term at which [`bessel_i0`] stops.
const BESSEL_I0_TOLERANCE: f64 = 1e-17;

/// Modified Bessel function of the first kind, order 0 (power series `Σ ((x/2)^k / k!)²`).
pub fn bessel_i0(x: f64) -> f64 {
    let half_sq = (x / 2.0) * (x / 2.0);
    let (mut sum, mut term, mut k) = (1.0f64, 1.0f64, 1.0f64);
    loop {
        term *= half_sq / (k * k);
        sum += term;
        if term < BESSEL_I0_TOLERANCE * sum {
            return sum;
        }
        k += 1.0;
    }
}

/// Symmetric Kaiser window of length `len` and shape `beta` (`scipy.signal.windows.kaiser(len, beta,
/// sym=True)`), in f64 for filter design.
pub fn kaiser_window(len: usize, beta: f64) -> Vec<f64> {
    if len <= 1 {
        return vec![1.0; len];
    }
    let denom = bessel_i0(beta);
    let last = (len - 1) as f64;
    (0..len)
        .map(|n| {
            let r = 2.0 * n as f64 / last - 1.0;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windows_symmetry_and_bounds() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-6);
        assert!(sinc(1.0).abs() < 1e-6);

        let g = gaussian_window(11, 2.0);
        assert_eq!(g.len(), 11);
        assert!((g[5] - 1.0).abs() < 1e-6);
        assert!((g[0] - g[10]).abs() < 1e-6);

        let h = hann_window(9);
        assert!(h[0].abs() < 1e-6);
        assert!((h[4] - 1.0).abs() < 1e-6);
    }
}
