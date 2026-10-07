//! Fractional-delay interpolation: `x(t + shift)` for `|shift| ≤ ½` from the `2·radius + 1` samples
//! around `t`, weighted by a Blackman-Harris windowed sinc normalized to sum to 1 (so constants pass
//! unchanged). Host functions and device building blocks share the same weights.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::kernels::fractional_delay_taps_kernel;
pub use super::kernels::windowed_sinc_weight;
use crate::core::{buffer, DspFloat};
use crate::math::windows::{blackman_window, sinc};

/// Weights `w[k + radius]` for source offsets `k = −radius..=radius` of the value at `+shift`
/// (`τ = k − shift`, `w ∝ sinc(τ) · BlackmanHarris(τ / radius)`), summing to 1.
pub fn fractional_delay_taps(shift: f32, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let raw: Vec<f32> = (-r..=r)
        .map(|k| {
            let tau = k as f32 - shift;
            sinc(tau) * blackman_window(tau, radius as f32)
        })
        .collect();
    let norm: f32 = raw.iter().sum();
    raw.into_iter().map(|w| w / norm).collect()
}

/// `out[i] = x(start + i + shift)` interpolated from `row`; every output uses its full
/// `2·radius + 1` taps, which must lie inside `row`.
pub fn fractional_delay(row: &[f32], start: usize, shift: f32, radius: usize, out: &mut [f32]) {
    assert!(start >= radius && start + out.len() + radius <= row.len(), "fractional-delay taps leave the row");
    let taps = fractional_delay_taps(shift, radius);
    for (i, o) in out.iter_mut().enumerate() {
        let base = start + i - radius;
        *o = taps.iter().zip(&row[base..]).map(|(w, x)| w * x).sum();
    }
}

/// Device weights (`[count, 2·radius + 1]` of `F`) of the `count` shifts in `shifts`.
pub fn fractional_delay_taps_device<F: DspFloat>(client: &Client, shifts: &Handle, count: usize, radius: usize) -> Handle {
    let n = 2 * radius + 1;
    let taps = buffer::empty::<F>(client, count * n);
    if count > 0 {
        let geom = LaunchGeometry::elementwise(client, count);
        // SAFETY: `shifts` holds `count` values of `F`; `taps` was just sized for `count · n`
        unsafe {
            fractional_delay_taps_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(shifts.clone(), count),
                BufferArg::from_raw_parts(taps.clone(), count * n),
                count as u32,
                radius as u32,
            );
        }
    }
    taps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_interpolate_a_line_and_pass_integers() {
        let taps = fractional_delay_taps(0.0, 5);
        assert!((taps[5] - 1.0).abs() < 1e-6 && taps.iter().enumerate().all(|(i, w)| i == 5 || w.abs() < 1e-6));
        // A constant passes unchanged; a slow sine is delayed by the shift
        let row: Vec<f32> = (0..64).map(|t| (t as f32 * 0.1).sin()).collect();
        let mut out = vec![0.0f32; 10];
        fractional_delay(&row, 20, 0.3, 5, &mut out);
        for (i, o) in out.iter().enumerate() {
            assert!((o - (((20 + i) as f32 + 0.3) * 0.1).sin()).abs() < 1e-3);
        }
    }

    fn device_taps_match_host(client: &Client) {
        let shifts = [-0.5f32, -0.2, 0.0, 0.37];
        let taps = buffer::download::<f32>(client, fractional_delay_taps_device::<f32>(client, &buffer::upload(client, &shifts), 4, 5));
        for (i, &s) in shifts.iter().enumerate() {
            for (d, h) in taps[i * 11..(i + 1) * 11].iter().zip(fractional_delay_taps(s, 5)) {
                assert!((d - h).abs() < 1e-5, "shift {s}: {d} vs {h}");
            }
        }
    }

    runtime_test!(device_taps_equal_host_taps, device_taps_match_host);
}
