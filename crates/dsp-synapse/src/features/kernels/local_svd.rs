//! Kernels of [`crate::features::local_svd`].

use cubecl::prelude::*;

/// Padding of a neighbourhood row: no channel.
pub const NO_CHANNEL: u32 = u32::MAX;

/// Sample `t` of `data` row `ch` (`[channels, samples]`), the nearest edge sample outside.
#[cube]
fn at<F: Float>(data: &[F], ch: u32, t: i32, samples: u32) -> F {
    let mut s = 0u32;
    if t > 0i32 {
        s = u32::min(t as u32, samples - 1u32);
    }
    data[(ch * samples + s) as usize]
}

/// One unit per fit peak `i`: its waveform on its own channel, `n_before` samples before the peak
/// to `width − n_before` after, into row `i` of `rows` (`[peaks, width]`). Kept only when its
/// largest `|sample|` is at the peak (the first on ties, as `argmax`), then signed so the peak
/// sample is positive; otherwise the row is zeros and `valid[i] = 0`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gather_fit_waveforms_kernel<F: Float>(
    data: &[F],
    peak_samples: &[u32],
    peak_channels: &[u32],
    rows: &mut [F],
    valid: &mut [u32],
    n_peaks: u32,
    samples: u32,
    n_before: u32,
    width: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n_peaks {
        let ch = peak_channels[i as usize];
        let t0 = peak_samples[i as usize] as i32 - n_before as i32;
        // First index of the largest |sample|
        let mut best = 0u32;
        let mut best_abs = F::new(-1.0f32);
        let mut k = 0u32;
        while k < width {
            let v = F::abs(at::<F>(data, ch, t0 + k as i32, samples));
            if v > best_abs {
                best_abs = v;
                best = k;
            }
            k += 1u32;
        }
        let peak = at::<F>(data, ch, t0 + n_before as i32, samples);
        let keep = best == n_before && peak != F::new(0.0f32);
        // 0 drops the row; otherwise the sign that makes the peak sample positive
        let mut sign = F::new(0.0f32);
        if keep && peak > F::new(0.0f32) {
            sign = F::new(1.0f32);
        }
        if keep && peak < F::new(0.0f32) {
            sign = F::new(-1.0f32);
        }
        let mut k = 0u32;
        while k < width {
            rows[(i * width + k) as usize] = sign * at::<F>(data, ch, t0 + k as i32, samples);
            k += 1u32;
        }
        valid[i as usize] = if keep { 1u32 } else { 0u32 };
    }
}

/// One unit per `(peak p, component c, local channel w)`: the projection of peak `p`'s waveform on
/// channel `neighbours[peak_channels[p] · max_neighbours + w]` onto component `c`,
/// `Σ_t data[ch, t0 + t] · components[c, t]`, into `features[(p · n_components + c) ·
/// max_neighbours + w]`; 0 where the neighbourhood row is padded.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn project_local_kernel<F: Float>(
    data: &[F],
    peak_samples: &[u32],
    peak_channels: &[u32],
    neighbours: &[u32],
    components: &[F],
    features: &mut [F],
    n_peaks: u32,
    samples: u32,
    n_before: u32,
    width: u32,
    n_components: u32,
    max_neighbours: u32,
) {
    let q = ABSOLUTE_POS as u32;
    if q < n_peaks * n_components * max_neighbours {
        let w = q % max_neighbours;
        let c = (q / max_neighbours) % n_components;
        let p = q / (max_neighbours * n_components);
        let ch = neighbours[(peak_channels[p as usize] * max_neighbours + w) as usize];
        let mut acc = F::new(0.0f32);
        if ch != NO_CHANNEL {
            let t0 = peak_samples[p as usize] as i32 - n_before as i32;
            let mut t = 0u32;
            while t < width {
                acc += at::<F>(data, ch, t0 + t as i32, samples) * components[(c * width + t) as usize];
                t += 1u32;
            }
        }
        features[q as usize] = acc;
    }
}
