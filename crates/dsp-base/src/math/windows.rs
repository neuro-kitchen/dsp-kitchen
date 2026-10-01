use std::f32::consts::PI;

/// Computes normalized sinc: $\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$, with $\text{sinc}(0) = 1$.
#[inline]
pub fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-7 {
        1.0
    } else {
        let pix = PI * x;
        pix.sin() / pix
    }
}

/// Blackman-Harris 4-term apodization window evaluated at `x` over `[-half_width, half_width]`.
#[inline]
pub fn blackman_window(x: f32, half_width: f32) -> f32 {
    let norm = (x / half_width.max(1e-12)).clamp(-1.0, 1.0);
    let u = (norm + 1.0) * 0.5;
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;

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
    let inv_two_sigma2 = 1.0 / (2.0 * sigma.max(1e-6).powi(2));
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
        .map(|i| 0.54 - 0.46 * (2.0 * PI * (i as f32) / denom).cos())
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
