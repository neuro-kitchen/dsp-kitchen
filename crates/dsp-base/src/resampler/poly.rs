//! Rational resampling (`scipy.signal.resample_poly`).

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::design::{firwin, FirWindow};
use super::kernels::upfirdn_kernel;
use crate::core::{buffer, cast_all, DspFloat, EdgeMode};

/// `resample_poly`'s padding (`padtype="constant"`, zeros).
pub const RESAMPLE_POLY_DEFAULT_EDGE: EdgeMode = EdgeMode::Zeros;

/// Window `resample_poly` designs its filter with by default.
pub const RESAMPLE_POLY_WINDOW: FirWindow = FirWindow::Kaiser { beta: 5.0 };

/// Half-length of the designed filter per unit of the larger rate (`half_len = 10 · max(up, down)`).
pub const RESAMPLE_POLY_HALF_LEN_PER_RATE: usize = 10;

/// The anti-aliasing filter of [`resample_poly`].
#[derive(Debug, Clone, PartialEq)]
pub enum ResampleFilter {
    /// Designed with `firwin(2 · half_len + 1, 1 / max(up, down), window)`.
    Window(FirWindow),
    /// Explicit taps at the up-sampled rate (odd length; centred).
    Taps(Vec<f64>),
}

impl Default for ResampleFilter {
    fn default() -> Self {
        ResampleFilter::Window(RESAMPLE_POLY_WINDOW)
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Samples out of `samples` resampled by `up / down` (`ceil(samples · up / down)`).
pub fn resample_poly_len(samples: usize, up: usize, down: usize) -> usize {
    let g = gcd(up, down).max(1);
    (samples * (up / g)).div_ceil(down / g)
}

/// Resamples every channel of a `[channels, samples]` buffer of `F` by `up / down` into `output`
/// (`[channels, resample_poly_len(samples, up, down)]`), as `scipy.signal.resample_poly`: the filter is
/// scaled by `up` and aligned so output sample 0 sits on input sample 0. Returns the output length.
///
/// # Panics
/// If `up` or `down` is zero.
#[allow(clippy::too_many_arguments)]
pub fn resample_poly<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    channels: usize,
    samples: usize,
    up: usize,
    down: usize,
    filter: &ResampleFilter,
    edge: EdgeMode,
) -> usize {
    assert!(up > 0 && down > 0, "resample_poly factors must be positive");
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    let out_len = resample_poly_len(samples, up, down);
    if channels == 0 || out_len == 0 {
        return out_len;
    }

    let (mut taps, half_len) = match filter {
        ResampleFilter::Window(window) => {
            let half_len = RESAMPLE_POLY_HALF_LEN_PER_RATE * up.max(down);
            (firwin(2 * half_len + 1, 1.0 / up.max(down) as f64, *window), half_len)
        }
        ResampleFilter::Taps(taps) => (taps.clone(), (taps.len().max(1) - 1) / 2),
    };
    for t in &mut taps {
        *t *= up as f64;
    }
    // scipy's alignment: pad the filter by `pre_pad` zeros and drop the first `pre_remove` outputs
    let pre_pad = down - half_len % down;
    let pre_remove = (half_len + pre_pad) / down;
    let start = (pre_remove * down) as i64 - pre_pad as i64;
    // Farthest the taps reach outside the input, for non-zero edge modes
    let s_min = (start - (taps.len() as i64 - 1)).div_euclid(up as i64);
    let s_max = ((out_len as i64 - 1) * down as i64 + start).div_euclid(up as i64);
    let ext_pad = (-s_min).max(s_max - (samples as i64 - 1)).max(0) as usize;

    let taps_h = buffer::upload(client, &cast_all::<F>(&taps));
    let geom = LaunchGeometry::channels_samples(client, channels, out_len);
    unsafe {
        upfirdn_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), channels * samples),
            BufferArg::from_raw_parts(taps_h, taps.len()),
            BufferArg::from_raw_parts(output.clone(), channels * out_len),
            channels as u32,
            samples as u32,
            out_len as u32,
            taps.len() as u32,
            up as u32,
            down as u32,
            start as i32,
            ext_pad as u32,
            edge.id(),
        );
    }
    out_len
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host `resample_poly` straight from its definition: up-sample with zeros, convolve with the
    /// padded filter, down-sample, keep `[pre_remove, pre_remove + out_len)`.
    fn host_resample_poly(x: &[f64], up: usize, down: usize) -> Vec<f64> {
        let half_len = RESAMPLE_POLY_HALF_LEN_PER_RATE * up.max(down);
        let mut h = firwin(2 * half_len + 1, 1.0 / up.max(down) as f64, RESAMPLE_POLY_WINDOW);
        h.iter_mut().for_each(|t| *t *= up as f64);
        let pre_pad = down - half_len % down;
        let pre_remove = (half_len + pre_pad) / down;
        let h: Vec<f64> = std::iter::repeat_n(0.0, pre_pad).chain(h).collect();
        let xu: Vec<f64> = (0..x.len() * up).map(|i| if i % up == 0 { x[i / up] } else { 0.0 }).collect();
        let out_len = resample_poly_len(x.len(), up, down);
        (0..out_len)
            .map(|m| {
                let n = (m + pre_remove) * down;
                (0..h.len()).filter(|&k| k <= n && n - k < xu.len()).map(|k| h[k] * xu[n - k]).sum()
            })
            .collect()
    }

    fn matches_host(client: &Client) {
        let samples = 997;
        let x: Vec<f64> = (0..samples).map(|i| (i as f64 * 0.05).sin() * 10.0 + ((i * 7919) % 31) as f64 * 0.1).collect();
        let input = buffer::upload(client, &x.iter().map(|v| *v as f32).collect::<Vec<_>>());
        for (up, down) in [(1usize, 3usize), (3, 1), (2, 3), (160, 147)] {
            let out_len = resample_poly_len(samples, up, down);
            let output = buffer::empty::<f32>(client, out_len);
            let n = resample_poly::<f32>(client, &input, &output, 1, samples, up, down, &ResampleFilter::default(), RESAMPLE_POLY_DEFAULT_EDGE);
            assert_eq!(n, out_len);
            let got = buffer::download::<f32>(client, output);
            let want = host_resample_poly(&x, up, down);
            for (m, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((*g as f64 - w).abs() < 1e-3, "{} {up}/{down} sample {m}: {g} vs {w}", client.name());
            }
        }
    }
    runtime_test!(test_resample_poly_matches_definition, matches_host);
}
