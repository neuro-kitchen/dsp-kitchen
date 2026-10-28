//! Windowed-sinc low-pass FIR design (`scipy.signal.firwin`), in f64.

use crate::math::kaiser_window;

/// Window applied to the ideal low-pass response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirWindow {
    /// Symmetric Hamming (`firwin`'s default).
    Hamming,
    /// Symmetric Kaiser of shape `beta` (`resample_poly` uses `beta = 5`).
    Kaiser { beta: f64 },
}

impl FirWindow {
    /// The window's `len` symmetric values.
    pub fn values(self, len: usize) -> Vec<f64> {
        match self {
            FirWindow::Hamming if len > 1 => {
                let last = (len - 1) as f64;
                (0..len).map(|n| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / last).cos()).collect()
            }
            FirWindow::Hamming => vec![1.0; len],
            FirWindow::Kaiser { beta } => kaiser_window(len, beta),
        }
    }
}

/// Normalized sinc `sin(πx) / (πx)`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) }
}

/// Low-pass FIR of `num_taps` taps with cutoff `cutoff` (fraction of Nyquist, `0 < cutoff ≤ 1`),
/// windowed by `window` and scaled to unit DC gain (`firwin(num_taps, cutoff, window=…)`).
pub fn firwin(num_taps: usize, cutoff: f64, window: FirWindow) -> Vec<f64> {
    assert!(num_taps > 0, "firwin needs at least one tap");
    assert!(cutoff > 0.0 && cutoff <= 1.0, "cutoff {cutoff} must lie in (0, 1] of Nyquist");
    let alpha = 0.5 * (num_taps - 1) as f64;
    let mut h: Vec<f64> = (0..num_taps).map(|n| cutoff * sinc(cutoff * (n as f64 - alpha))).collect();
    for (v, w) in h.iter_mut().zip(window.values(num_taps)) {
        *v *= w;
    }
    let dc: f64 = h.iter().sum();
    for v in &mut h {
        *v /= dc;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn firwin_is_symmetric_with_unit_dc_gain() {
        for window in [FirWindow::Hamming, FirWindow::Kaiser { beta: 5.0 }] {
            let h = firwin(41, 0.25, window);
            assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            for k in 0..20 {
                assert!((h[k] - h[40 - k]).abs() < 1e-15);
            }
            assert!(h[20] > h[19] && h[20] > 0.2);
        }
    }

    #[test]
    fn kaiser_window_shape() {
        let w = kaiser_window(11, 5.0);
        assert!((w[5] - 1.0).abs() < 1e-12);
        assert!((w[0] - w[10]).abs() < 1e-15);
        // scipy.signal.windows.kaiser(11, 5.0)[0] = 1 / I0(5) = 0.03671089
        assert!((w[0] - 0.036_710_892).abs() < 1e-8);
    }
}
