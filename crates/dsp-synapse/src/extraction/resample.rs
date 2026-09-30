use std::f32::consts::PI;

/// Computes normalized sinc: sinc(x) = sin(pi * x) / (pi * x), sinc(0) = 1.
#[inline]
pub fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-7 {
        1.0
    } else {
        let pix = PI * x;
        pix.sin() / pix
    }
}

/// Blackman-Harris 4-term apodization window to prevent spectral leakage in sinc resampling.
#[inline]
pub fn blackman_window(x: f32, half_width: f32) -> f32 {
    let norm = (x / half_width).clamp(-1.0, 1.0);
    let u = (norm + 1.0) * 0.5; // [0, 1]
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;

    let two_pi_u = 2.0 * PI * u;
    a0 - a1 * two_pi_u.cos() + a2 * (2.0 * two_pi_u).cos() - a3 * (3.0 * two_pi_u).cos()
}

/// Half-width of the windowed-sinc kernel used for snippet realignment (taps `−5..=5`).
pub const SINC_KERNEL_RADIUS: usize = 5;

/// Windowed-sinc tap weight for source offset `k` when evaluating at fractional `shift`.
#[inline]
pub fn sinc_tap(k: isize, shift: f32, radius: usize) -> f32 {
    let tau = k as f32 - shift;
    sinc(tau) * blackman_window(tau, radius as f32)
}

/// Writes `out[i] = x(start + i + shift)` interpolated from `row` with the full `2·radius + 1`
/// windowed-sinc taps read from `row` itself (weights normalized to sum to 1).
///
/// Realignment convention: a trough found at `center + δ` ([`super::parabolic_subsample_offset`])
/// lands on `center` with `shift = +δ`.
///
/// # Panics
/// If `start < radius` or `start + out.len() + radius > row.len()`.
pub fn interpolate_window(row: &[f32], start: usize, shift: f32, radius: usize, out: &mut [f32]) {
    assert!(start >= radius && start + out.len() + radius <= row.len(), "sinc taps leave the row");
    let r = radius as isize;
    let weights: Vec<f32> = (-r..=r).map(|k| sinc_tap(k, shift, radius)).collect();
    let norm: f32 = weights.iter().sum();
    for (i, o) in out.iter_mut().enumerate() {
        let base = start + i - radius;
        let sum: f32 = weights.iter().zip(&row[base..]).map(|(w, x)| w * x).sum();
        *o = sum / norm;
    }
}

/// Applies sub-sample fractional temporal realignment to a 1D waveform using windowed sinc
/// interpolation: `out[i] = x(i + shift_samples)`. Near the ends the kernel is truncated and
/// renormalized; for snippets cut from a longer trace prefer [`interpolate_window`].
///
/// - `waveform`: Input samples
/// - `shift_samples`: Shift in samples (`+δ` moves a trough at `i + δ` onto `i`)
/// - `kernel_radius`: Window half-width (default: 4 to 6 samples)
pub fn resample_sinc_1d(waveform: &[f32], shift_samples: f32, kernel_radius: usize) -> Vec<f32> {
    if waveform.is_empty() || shift_samples.abs() < 1e-5 {
        return waveform.to_vec();
    }

    let n = waveform.len();
    let mut aligned = vec![0.0f32; n];
    let radius = kernel_radius as isize;
    let half_w_f32 = kernel_radius as f32;

    for i in 0..n {
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        for k in -radius..=radius {
            let src_idx = (i as isize) + k;
            if src_idx >= 0 && (src_idx as usize) < n {
                let tau = k as f32 - shift_samples;
                let w = blackman_window(tau, half_w_f32);
                let s = sinc(tau);
                let weight = s * w;

                sum += waveform[src_idx as usize] * weight;
                weight_sum += weight;
            }
        }

        aligned[i] = if weight_sum.abs() > 1e-6 {
            sum / weight_sum
        } else {
            waveform[i]
        };
    }

    aligned
}

/// Applies sub-sample fractional temporal realignment across a 2D multi-channel snippet [channels, samples].
pub fn resample_sinc_multichannel(
    waveform: &[f32],
    channels: usize,
    samples: usize,
    shift_samples: f32,
    kernel_radius: usize,
) -> Vec<f32> {
    assert_eq!(waveform.len(), channels * samples);
    let mut aligned = vec![0.0f32; channels * samples];

    for ch in 0..channels {
        let offset = ch * samples;
        let ch_slice = &waveform[offset..offset + samples];
        let ch_aligned = resample_sinc_1d(ch_slice, shift_samples, kernel_radius);
        aligned[offset..offset + samples].copy_from_slice(&ch_aligned);
    }

    aligned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinc_resampling_shift() {
        // Gaussian peak centered at sample 10.3
        let n = 20;
        let mut raw = vec![0.0f32; n];
        for i in 0..n {
            let t = (i as f32 - 10.3) / 2.0;
            raw[i] = (-0.5 * t * t).exp() * -100.0;
        }

        // Evaluating y(t + 0.3) puts the peak from 10.3 onto sample 10.0
        let aligned = resample_sinc_1d(&raw, 0.3, 5);

        // Minimum should now be at sample 10
        let (min_idx, _) = aligned
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        assert_eq!(min_idx, 10);
    }
}
