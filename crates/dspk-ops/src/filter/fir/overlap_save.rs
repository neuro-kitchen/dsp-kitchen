//! FIR filtering by FFT (overlap-save) for long filters: O(samples · log N) instead of
//! O(samples · taps). Direct convolution wins for short filters (the direct, tiled and vectorized
//! kernels); the FIR autotuner offers this one as a candidate from [`MIN_TAPS`] taps up and keeps
//! whichever is faster on the device (Adámek et al., "GPU fast convolution via the overlap-and-save
//! method in shared memory").
//!
//! Both FIR shapes become one causal convolution over the row extended by its edge mode (the same
//! `read_extended` the direct kernels read):
//! - causal `y[t] = Σ taps[k] · x[t − k]`: the row padded `L − 1` on the left, filter `taps`;
//! - centred `y[t] = Σ taps[k] · x[t + k − r]`: the row padded `r` on each side, filter `taps`
//!   reversed.
//!
//! The extended row is cut into frames of `N` (a power of two ≥ 4 · taps) that overlap by
//! `L − 1`; each frame is transformed (`cubek-fft` `rfft`), multiplied by the filter's spectrum,
//! transformed back (`irfft`, normalized), and its last `N − L + 1` samples are outputs.
//!
//! Float32 only (`cubek-fft`); equal to the direct kernels within FFT rounding, not bit for bit.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::std::tensor::TensorHandle;
use dspk_core::compute::{element_position, LaunchGeometry};

use crate::core::kernels::edge::read_extended;
use crate::core::{buffer, DspFloat, EdgeMode};

/// Taps from which the FFT path is worth timing against the direct kernels.
pub const MIN_TAPS: usize = 64;

/// The FFT length for a filter of `num_taps`: a power of two ≥ 4 × taps, at least 512, at most
/// 16 384 (shorter frames waste less on the overlap; longer ones amortize the transforms).
pub fn fft_len(num_taps: usize) -> usize {
    (4 * num_taps).next_power_of_two().clamp(512, 16_384)
}

/// Whether the FFT path can run `num_taps` taps for element type `F`.
pub fn supports<F: DspFloat>(num_taps: usize) -> bool {
    size_of::<F>() == 4 && num_taps >= 2 && 4 * num_taps <= 16_384
}

/// `output = fir(input)` by overlap-save: `channels` rows of `samples`, `num_taps` taps (`taps` on
/// the device), causal or centred (then `num_taps = 2 · radius + 1`), edges by `edge`.
#[allow(clippy::too_many_arguments)]
pub fn execute<F: DspFloat>(client: &Client, input: &Handle, output: &Handle, taps: &Handle, channels: usize, samples: usize, num_taps: usize, centered: bool, edge: EdgeMode) {
    debug_assert!(supports::<F>(num_taps));
    if channels == 0 || samples == 0 {
        return;
    }
    let dtype = ElemType::Float(FloatKind::F32);
    let n_fft = fft_len(num_taps);
    let step = n_fft - (num_taps - 1);
    let (pad, right) = if centered { ((num_taps - 1) / 2, (num_taps - 1) / 2) } else { (num_taps - 1, 0) };
    let frames_per_row = samples.div_ceil(step);
    let rows = channels * frames_per_row;

    // The filter on N samples (reversed for a centred one), and its spectrum
    let padded_taps = buffer::empty::<F>(client, n_fft);
    let geom = LaunchGeometry::elementwise(client, n_fft);
    // SAFETY: `taps` holds `num_taps` values, `padded_taps` `n_fft`
    unsafe {
        ols_taps_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(taps.clone(), num_taps),
            BufferArg::from_raw_parts(padded_taps.clone(), n_fft),
            num_taps as u32,
            n_fft as u32,
            centered,
        );
    }
    let tensor = |handle: Handle, shape: Vec<usize>| TensorHandle::new_contiguous(shape, handle, dtype);
    let (h_re, h_im) = cubek_fft::rfft(client, tensor(padded_taps, vec![1, n_fft]), 1, dtype);

    // The extended rows cut into overlapping frames
    let frames = buffer::empty::<F>(client, rows * n_fft);
    let total = rows * n_fft;
    let geom = LaunchGeometry::elementwise(client, total);
    // SAFETY: `input` holds `channels · samples` values, `frames` `total`
    unsafe {
        ols_gather_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), channels * samples),
            BufferArg::from_raw_parts(frames.clone(), total),
            samples as u32,
            pad as u32,
            (samples + pad + right) as u32,
            step as u32,
            n_fft as u32,
            frames_per_row as u32,
            total as u32,
            edge.id(),
        );
    }
    let (re, im) = cubek_fft::rfft(client, tensor(frames, vec![rows, n_fft]), 1, dtype);
    let bins = n_fft / 2 + 1;
    let spectrum = rows * bins;
    let geom = LaunchGeometry::elementwise(client, spectrum);
    // SAFETY: the frames' spectra hold `rows · bins` values each, the filter's `bins`
    unsafe {
        ols_multiply_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(re.handle.clone(), spectrum),
            BufferArg::from_raw_parts(im.handle.clone(), spectrum),
            BufferArg::from_raw_parts(h_re.handle.clone(), bins),
            BufferArg::from_raw_parts(h_im.handle.clone(), bins),
            bins as u32,
            spectrum as u32,
        );
    }
    let filtered = cubek_fft::irfft(client, re, im, 1, dtype);

    // Each frame's last `step` samples are outputs
    let out_total = channels * samples;
    let geom = LaunchGeometry::elementwise(client, out_total);
    // SAFETY: `filtered` holds `rows · n_fft` values, `output` `channels · samples`
    unsafe {
        ols_scatter_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(filtered.handle.clone(), total),
            BufferArg::from_raw_parts(output.clone(), out_total),
            samples as u32,
            step as u32,
            n_fft as u32,
            frames_per_row as u32,
            (num_taps - 1) as u32,
            out_total as u32,
        );
    }
}

/// `padded[i]`: the filter for a causal convolution (`taps` as they are, or reversed when
/// `centered`), zero past `num_taps`.
#[cube(launch)]
fn ols_taps_kernel<F: Float>(taps: &[F], padded: &mut [F], num_taps: u32, n_fft: u32, #[comptime] centered: bool) {
    let i = element_position() as u32;
    if i < n_fft {
        let mut v = F::new(0.0f32);
        if i < num_taps {
            if comptime!(centered) {
                v = taps[(num_taps - 1u32 - i) as usize];
            } else {
                v = taps[i as usize];
            }
        }
        padded[i as usize] = v;
    }
}

/// Frame `f` of row `c` holds extended samples `f · step ..+ n_fft` (zero past `limit`).
#[cube(launch)]
fn ols_gather_kernel<F: Float>(
    input: &[F],
    frames: &mut [F],
    samples: u32,
    pad: u32,
    limit: u32,
    step: u32,
    n_fft: u32,
    frames_per_row: u32,
    total: u32,
    #[comptime] edge: u32,
) {
    let i = element_position() as u32;
    if i < total {
        let n = i % n_fft;
        let row = i / n_fft;
        let f = row % frames_per_row;
        let c = row / frames_per_row;
        let u = f * step + n;
        let mut v = F::new(0.0f32);
        if u < limit {
            v = read_extended::<F>(input, (c * samples) as usize, samples, pad, u, edge);
        }
        frames[i as usize] = v;
    }
}

/// Each frame's spectrum times the filter's (complex).
#[cube(launch)]
fn ols_multiply_kernel<F: Float>(re: &mut [F], im: &mut [F], h_re: &[F], h_im: &[F], bins: u32, total: u32) {
    let i = element_position() as u32;
    if i < total {
        let k = (i % bins) as usize;
        let (a, b) = (re[i as usize], im[i as usize]);
        let (c, d) = (h_re[k], h_im[k]);
        re[i as usize] = a * c - b * d;
        im[i as usize] = a * d + b * c;
    }
}

/// Output `t` of row `c` is sample `skip + t % step` of frame `t / step`.
#[cube(launch)]
fn ols_scatter_kernel<F: Float>(filtered: &[F], output: &mut [F], samples: u32, step: u32, n_fft: u32, frames_per_row: u32, skip: u32, total: u32) {
    let i = element_position() as u32;
    if i < total {
        let c = i / samples;
        let t = i % samples;
        let f = t / step;
        output[i as usize] = filtered[((c * frames_per_row + f) * n_fft + skip + t % step) as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::fir::{execute_fir_centered_with, execute_fir_with, FirKernel};
    use dspk_core::compute::{ComputeTarget, ComputeTask};

    struct MatchesDirect;

    impl ComputeTask for MatchesDirect {
        type Output = ();
        fn run(self, client: Client) {
            let (channels, samples) = (3usize, 20_011usize);
            let x: Vec<f32> = (0..channels * samples).map(|i| ((i % samples) as f32 * 0.013).sin() * 50.0 + ((i * 7919) % 1013) as f32 * 0.05 + (i / samples) as f32 * 10.0).collect();
            let input = buffer::upload(&client, &x);
            for &(num_taps, centered) in &[(65usize, false), (129, true), (300, false), (201, true)] {
                let taps_host: Vec<f32> = (0..num_taps).map(|k| ((k as f32 * 0.37).cos() * 0.8) / num_taps as f32).collect();
                let taps = buffer::upload(&client, &taps_host);
                for edge in [EdgeMode::Zeros, EdgeMode::Odd, EdgeMode::Reflect, EdgeMode::Nearest] {
                    let (direct, fft) = (buffer::empty::<f32>(&client, channels * samples), buffer::empty::<f32>(&client, channels * samples));
                    if centered {
                        execute_fir_centered_with::<f32>(&client, &input, &direct, &taps, channels, samples, num_taps / 2, edge, Some(FirKernel::Direct));
                    } else {
                        execute_fir_with::<f32>(&client, &input, &direct, &taps, channels, samples, num_taps, edge, Some(FirKernel::Direct));
                    }
                    execute::<f32>(&client, &input, &fft, &taps, channels, samples, num_taps, centered, edge);
                    let (a, b): (Vec<f32>, Vec<f32>) = (buffer::download(&client, direct), buffer::download(&client, fft));
                    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
                    let worst = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
                    assert!(worst <= 2e-5 * scale, "{} taps {num_taps} centred {centered} {edge:?}: {worst} of {scale}", client.name());
                }
            }
        }
    }

    #[test]
    fn overlap_save_equals_direct_convolution() {
        for target in ComputeTarget::for_tests() {
            target.run(MatchesDirect).expect("compiled-in runtime");
        }
    }
}
