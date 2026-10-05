//! Channel covariance `C = (X − μ)(X − μ)ᵀ / samples` of a `[channels, samples]` device buffer.
//!
//! Three launches: per-channel means ([`reduce::row_mean_std`]), partial sums of every channel pair
//! over sample splits (one unit per `(pair, split)`), then the sum of the splits per pair written to
//! both halves of the symmetric matrix. Splitting the samples keeps every unit busy for small
//! channel counts and bounds how many terms any single sum adds.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use crate::core::{buffer, reduce, DspFloat};

/// Samples one unit sums before the splits are merged.
pub const COVARIANCE_SPLIT_SAMPLES: usize = 4096;

/// `partial[split · pairs + p] = Σ_t (x[i, t] − μ_i)(x[j, t] − μ_j)` over the split's samples, for
/// the pair `(pair_i[p], pair_j[p])`. One unit per `(pair, split)` (`ABSOLUTE_POS`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn covariance_partial_kernel<F: Float>(
    input: &Array<F>,
    means: &Array<F>,
    pair_i: &Array<u32>,
    pair_j: &Array<u32>,
    partial: &mut Array<F>,
    samples: u32,
    pairs: u32,
    splits: u32,
    split_len: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < pairs * splits {
        let split = unit / pairs;
        let p = unit - split * pairs;
        let (i, j) = (pair_i[p as usize], pair_j[p as usize]);
        let (mi, mj) = (means[i as usize], means[j as usize]);
        let (base_i, base_j) = ((i * samples) as usize, (j * samples) as usize);
        let end = u32::min((split + 1u32) * split_len, samples);
        let mut acc = F::new(0.0f32);
        let mut t = split * split_len;
        while t < end {
            acc += (input[base_i + t as usize] - mi) * (input[base_j + t as usize] - mj);
            t += 1u32;
        }
        partial[unit as usize] = acc;
    }
}

/// Sums the splits of every pair, divides by `samples` and writes `cov[i, j]` and `cov[j, i]`.
/// One unit per pair.
#[cube(launch)]
pub fn covariance_merge_kernel<F: Float>(
    partial: &Array<F>,
    pair_i: &Array<u32>,
    pair_j: &Array<u32>,
    cov: &mut Array<F>,
    channels: u32,
    samples: u32,
    pairs: u32,
    splits: u32,
) {
    let p = ABSOLUTE_POS as u32;
    if p < pairs {
        let mut acc = F::new(0.0f32);
        let mut s = 0u32;
        while s < splits {
            acc += partial[(s * pairs + p) as usize];
            s += 1u32;
        }
        let value = acc / F::cast_from(u32::max(samples, 1u32));
        let (i, j) = (pair_i[p as usize], pair_j[p as usize]);
        cov[(i * channels + j) as usize] = value;
        cov[(j * channels + i) as usize] = value;
    }
}

/// The `i ≤ j` channel pairs, as two index arrays.
fn upper_pairs(channels: usize) -> (Vec<u32>, Vec<u32>) {
    (0..channels as u32).flat_map(|i| (i..channels as u32).map(move |j| (i, j))).unzip()
}

/// Covariance (`/ samples`) of a `[channels, samples]` buffer of `F` into `out_cov`
/// (`[channels, channels]`, row-major) and the channel means into `out_mean` (`channels` values).
pub fn covariance<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    input: &Handle,
    out_cov: &Handle,
    out_mean: &Handle,
    channels: usize,
    samples: usize,
) {
    if channels == 0 {
        return;
    }
    let std_scratch = buffer::empty::<R, F>(client, channels);
    reduce::row_mean_std::<R, F>(client, input, out_mean, &std_scratch, channels, samples);

    let (pi, pj) = upper_pairs(channels);
    let pairs = pi.len();
    let splits = samples.div_ceil(COVARIANCE_SPLIT_SAMPLES).max(1);
    let (pair_i, pair_j) = (buffer::upload(client, &pi), buffer::upload(client, &pj));
    let partial = buffer::empty::<R, F>(client, pairs * splits);

    let geom = LaunchGeometry::elementwise(client, pairs * splits);
    unsafe {
        covariance_partial_kernel::launch::<F, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), channels * samples),
            ArrayArg::from_raw_parts(out_mean.clone(), channels),
            ArrayArg::from_raw_parts(pair_i.clone(), pairs),
            ArrayArg::from_raw_parts(pair_j.clone(), pairs),
            ArrayArg::from_raw_parts(partial.clone(), pairs * splits),
            samples as u32,
            pairs as u32,
            splits as u32,
            COVARIANCE_SPLIT_SAMPLES as u32,
        );
    }
    let geom = LaunchGeometry::elementwise(client, pairs);
    unsafe {
        covariance_merge_kernel::launch::<F, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(partial, pairs * splits),
            ArrayArg::from_raw_parts(pair_i, pairs),
            ArrayArg::from_raw_parts(pair_j, pairs),
            ArrayArg::from_raw_parts(out_cov.clone(), channels * channels),
            channels as u32,
            samples as u32,
            pairs as u32,
            splits as u32,
        );
    }
}

/// Covariance and means of host data (`[channels, samples]`, uploaded as `F`): the covariance stays
/// on the device (ready for [`crate::linalg::symmetric_eigen`]), the means come back to the host.
pub fn covariance_of_host<R: Runtime, F: DspFloat>(client: &ComputeClient<R>, data: &[f32], channels: usize, samples: usize) -> (Handle, Vec<f64>) {
    assert_eq!(data.len(), channels * samples, "data size mismatch");
    let input = buffer::upload(client, &crate::core::cast_f32::<F>(data));
    let cov = buffer::empty::<R, F>(client, channels * channels);
    let mean = buffer::empty::<R, F>(client, channels);
    covariance::<R, F>(client, &input, &cov, &mean, channels, samples);
    let mean = buffer::download::<R, F>(client, mean).into_iter().map(crate::core::to_f64).collect();
    (cov, mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches_host<R: Runtime>(client: &ComputeClient<R>) {
        for (channels, samples) in [(1usize, 10usize), (5, 9_000), (17, 1_234)] {
            let data: Vec<f32> = (0..channels * samples)
                .map(|i| {
                    let (c, t) = ((i / samples) as f32, (i % samples) as f32);
                    (t * 0.013 * (c + 1.0)).sin() * 20.0 + ((i * 7919) % 101) as f32 * 0.1 + c * 50.0
                })
                .collect();
            let input = buffer::upload(client, &data);
            let (cov, mean) = (buffer::empty::<R, f32>(client, channels * channels), buffer::empty::<R, f32>(client, channels));
            covariance::<R, f32>(client, &input, &cov, &mean, channels, samples);
            let cov = buffer::download::<R, f32>(client, cov);

            let m: Vec<f64> = (0..channels).map(|c| data[c * samples..(c + 1) * samples].iter().map(|v| *v as f64).sum::<f64>() / samples as f64).collect();
            for i in 0..channels {
                for j in 0..channels {
                    let want: f64 = (0..samples)
                        .map(|t| (data[i * samples + t] as f64 - m[i]) * (data[j * samples + t] as f64 - m[j]))
                        .sum::<f64>()
                        / samples as f64;
                    let got = cov[i * channels + j] as f64;
                    assert!((got - want).abs() < 1e-3 * want.abs().max(1.0), "{} {channels}x{samples} [{i},{j}]: {got} vs {want}", R::name(client));
                }
            }
        }
    }
    runtime_test!(test_covariance_matches_host, matches_host);
}
