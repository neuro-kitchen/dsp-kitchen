//! Channel covariance `C = (X − μ)(X − μ)ᵀ / samples` of a `[channels, samples]` device buffer, and
//! the uncentred second moment `X Xᵀ / samples` accumulated over many buffers on the device
//! ([`SecondMomentAccumulator`]).
//!
//! Three launches: per-channel means ([`reduce::row_mean_std`]; skipped for the second moment),
//! partial sums of every channel pair over sample splits (one unit per `(pair, split)`), then the
//! sum of the splits per pair written to (or added to) both halves of the symmetric matrix.
//! Splitting the samples keeps every unit busy for small channel counts and bounds how many terms
//! any single sum adds.

use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use crate::core::{buffer, reduce, DspFloat};

/// Samples one unit sums before the splits are merged.
pub const COVARIANCE_SPLIT_SAMPLES: usize = 4096;

/// `partial[split · pairs + p] = Σ_t (x[i, t] − μ_i)(x[j, t] − μ_j)` over the split's samples, for
/// the pair `(pair_i[p], pair_j[p])`. Sample `t` of row `i` is `input[i · row_len + col_start + t]`
/// (`t < samples`). `centred = false` ignores `means`. One unit per `(pair, split)`
/// (`ABSOLUTE_POS`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn covariance_partial_kernel<F: Float>(
    input: &Array<F>,
    means: &Array<F>,
    pair_i: &Array<u32>,
    pair_j: &Array<u32>,
    partial: &mut Array<F>,
    row_len: u32,
    col_start: u32,
    samples: u32,
    pairs: u32,
    splits: u32,
    split_len: u32,
    #[comptime] centred: bool,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit < pairs * splits {
        let split = unit / pairs;
        let p = unit - split * pairs;
        let (i, j) = (pair_i[p as usize], pair_j[p as usize]);
        let mut mi = F::new(0.0f32);
        let mut mj = F::new(0.0f32);
        if centred {
            mi = means[i as usize];
            mj = means[j as usize];
        }
        let (base_i, base_j) = ((i * row_len + col_start) as usize, (j * row_len + col_start) as usize);
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

/// Sums the splits of every pair, divides by `samples` and writes `cov[i, j]` and `cov[j, i]`
/// (`accumulate = true` adds to them instead). One unit per pair.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn covariance_merge_kernel<F: Float>(
    partial: &Array<F>,
    pair_i: &Array<u32>,
    pair_j: &Array<u32>,
    cov: &mut Array<F>,
    channels: u32,
    samples: u32,
    pairs: u32,
    splits: u32,
    #[comptime] accumulate: bool,
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
        let (ij, ji) = ((i * channels + j) as usize, (j * channels + i) as usize);
        if accumulate {
            cov[ij] += value;
            if i != j {
                cov[ji] += value;
            }
        } else {
            cov[ij] = value;
            cov[ji] = value;
        }
    }
}

/// The `i ≤ j` channel pairs, as two index arrays.
fn upper_pairs(channels: usize) -> (Vec<u32>, Vec<u32>) {
    (0..channels as u32).flat_map(|i| (i..channels as u32).map(move |j| (i, j))).unzip()
}

/// Device pair index arrays for a channel count: the `i ≤ j` pairs of [`upper_pairs`].
struct PairIndex {
    pair_i: Handle,
    pair_j: Handle,
    pairs: usize,
}

impl PairIndex {
    fn new<R: Runtime>(client: &ComputeClient<R>, channels: usize) -> Self {
        let (pi, pj) = upper_pairs(channels);
        Self { pairs: pi.len(), pair_i: buffer::upload(client, &pi), pair_j: buffer::upload(client, &pj) }
    }
}

/// Where the pair sums read from: columns `cols` of every row of a `[channels, row_len]` buffer.
struct PairSumInput<'a> {
    input: &'a Handle,
    channels: usize,
    row_len: usize,
    cols: Range<usize>,
}

/// Sample splits of `samples` samples (one partial sum per pair and split).
fn splits_of(samples: usize) -> usize {
    samples.div_ceil(COVARIANCE_SPLIT_SAMPLES).max(1)
}

/// `out[i, j] (=|+=) Σ_{t ∈ cols} (x[i, t] − μ_i)(x[j, t] − μ_j) / |cols|` for every pair, centred on
/// `means` when given. `partial` holds at least `pairs · splits_of(|cols|)` values of `F`.
fn pair_sums<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    src: PairSumInput<'_>,
    means: Option<&Handle>,
    index: &PairIndex,
    partial: &Handle,
    out: &Handle,
    accumulate: bool,
) {
    let PairSumInput { input, channels, row_len, cols } = src;
    let samples = cols.len();
    let pairs = index.pairs;
    let splits = splits_of(samples);
    // Uncentred, the kernel never reads `means`: bind `out` (`F`, ≥ `channels` values), which
    // this launch does not write
    let centred = means.is_some();
    let means = means.unwrap_or(out).clone();

    let geom = LaunchGeometry::elementwise(client, pairs * splits);
    unsafe {
        covariance_partial_kernel::launch::<F, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(input.clone(), channels * row_len),
            ArrayArg::from_raw_parts(means, channels),
            ArrayArg::from_raw_parts(index.pair_i.clone(), pairs),
            ArrayArg::from_raw_parts(index.pair_j.clone(), pairs),
            ArrayArg::from_raw_parts(partial.clone(), pairs * splits),
            row_len as u32,
            cols.start as u32,
            samples as u32,
            pairs as u32,
            splits as u32,
            COVARIANCE_SPLIT_SAMPLES as u32,
            centred,
        );
    }
    let geom = LaunchGeometry::elementwise(client, pairs);
    unsafe {
        covariance_merge_kernel::launch::<F, R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(partial.clone(), pairs * splits),
            ArrayArg::from_raw_parts(index.pair_i.clone(), pairs),
            ArrayArg::from_raw_parts(index.pair_j.clone(), pairs),
            ArrayArg::from_raw_parts(out.clone(), channels * channels),
            channels as u32,
            samples as u32,
            pairs as u32,
            splits as u32,
            accumulate,
        );
    }
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
    let index = PairIndex::new(client, channels);
    let partial = buffer::empty::<R, F>(client, index.pairs * splits_of(samples));
    let src = PairSumInput { input, channels, row_len: samples, cols: 0..samples };
    pair_sums::<R, F>(client, src, Some(out_mean), &index, &partial, out_cov, false);
}

/// Mean over batches of the uncentred second moment `X Xᵀ / samples`, accumulated on the device:
/// every [`Self::add`] adds one batch (columns of a device buffer, e.g. the interior of a
/// halo-padded window) with equal weight, and only [`Self::finish`] downloads the
/// `[channels, channels]` result. This is the whitening covariance of Kilosort4 / EMUsort
/// (`CC += X Xᵀ / n` per batch, `CC / k`), for high-passed data whose mean is ~0.
pub struct SecondMomentAccumulator<R: Runtime, F: DspFloat> {
    client: ComputeClient<R>,
    channels: usize,
    index: PairIndex,
    sum: Handle,
    batches: usize,
    /// Partial sums, kept between batches; sized for `partial_samples` samples.
    partial: Handle,
    partial_samples: usize,
    _float: std::marker::PhantomData<F>,
}

impl<R: Runtime, F: DspFloat> SecondMomentAccumulator<R, F> {
    pub fn new(client: &ComputeClient<R>, channels: usize) -> Self {
        Self {
            client: client.clone(),
            channels,
            index: PairIndex::new(client, channels),
            sum: buffer::zeros::<R, F>(client, channels * channels),
            batches: 0,
            partial: buffer::empty::<R, F>(client, 1),
            partial_samples: 0,
            _float: std::marker::PhantomData,
        }
    }

    /// Adds columns `cols` of a `[channels, row_len]` device buffer of `F` as one batch. Empty
    /// column ranges are ignored.
    pub fn add(&mut self, input: &Handle, row_len: usize, cols: Range<usize>) {
        assert!(cols.end <= row_len, "columns {cols:?} outside rows of {row_len}");
        if self.channels == 0 || cols.is_empty() {
            return;
        }
        if cols.len() > self.partial_samples {
            self.partial = buffer::empty::<R, F>(&self.client, self.index.pairs * splits_of(cols.len()));
            self.partial_samples = cols.len();
        }
        let src = PairSumInput { input, channels: self.channels, row_len, cols };
        pair_sums::<R, F>(&self.client, src, None, &self.index, &self.partial, &self.sum, true);
        self.batches += 1;
    }

    /// Batches added so far.
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// The device `[channels, channels]` sum over batches of `X Xᵀ / n` (row-major, `F`), for
    /// device consumers such as [`crate::linalg::symmetric_eigen`]: nothing is read back.
    pub fn into_sum(self) -> Handle {
        self.sum
    }

    /// The `[channels, channels]` mean second moment (row-major); zeros before any batch.
    pub fn finish(self) -> Vec<f64> {
        let k = self.batches.max(1) as f64;
        buffer::download::<R, F>(&self.client, self.sum).into_iter().map(|v| crate::core::to_f64(v) / k).collect()
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

    /// Two batches of different lengths, read from the interior columns of padded rows.
    fn second_moment_matches_host<R: Runtime>(client: &ComputeClient<R>) {
        let channels = 3;
        let batches = [(40usize, 5..35usize), (25, 3..20)];
        let mut acc = SecondMomentAccumulator::<R, f32>::new(client, channels);
        let mut want = vec![0.0f64; channels * channels];
        for (b, (row_len, cols)) in batches.iter().enumerate() {
            let data: Vec<f32> = (0..channels * row_len).map(|i| ((i * 31 + b * 7) % 17) as f32 - 8.0).collect();
            acc.add(&buffer::upload(client, &data), *row_len, cols.clone());
            for i in 0..channels {
                for j in 0..channels {
                    let s: f64 = cols.clone().map(|t| data[i * row_len + t] as f64 * data[j * row_len + t] as f64).sum();
                    want[i * channels + j] += s / cols.len() as f64 / batches.len() as f64;
                }
            }
        }
        assert_eq!(acc.batches(), 2);
        for (k, (got, want)) in acc.finish().iter().zip(&want).enumerate() {
            assert!((got - want).abs() < 1e-3 * want.abs().max(1.0), "{} [{k}]: {got} vs {want}", R::name(client));
        }
    }
    runtime_test!(test_second_moment_matches_host, second_moment_matches_host);
}
