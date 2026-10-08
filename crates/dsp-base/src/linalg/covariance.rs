//! Channel covariance `C = (X − μ)(X − μ)ᵀ / samples` of a `[channels, samples]` device buffer, and
//! the uncentred second moment `X Xᵀ / samples` accumulated over many buffers on the device
//! ([`SecondMomentAccumulator`]).
//!
//! Both are products of the samples with themselves ([`fn@super::matmul`]). [`fn@split_rows_kernel`]
//! first copies the samples (centred for the covariance) into a split-major buffer `[splits,
//! channels, COVARIANCE_SPLIT_SAMPLES]`, the last split zero-padded; one batched product then gives
//! every split's `X_s X_sᵀ`: no single sum adds more than [`COVARIANCE_SPLIT_SAMPLES`] terms, and
//! few channels still give the device enough independent work. [`fn@sum_slices_kernel`] adds the
//! partial products, divides by the true sample count and writes (or adds to) the result.
//!
//! The copy is what keeps the product's batches outermost: a strided view whose batch stride is
//! smaller than its row stride (splits interleaved inside the rows) is computed wrongly by
//! cubek-matmul when the row stride is odd.

use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::kernels::{split_rows_kernel, sum_slices_kernel};
use super::matmul::{matmul, MatrixView};
use crate::core::{buffer, cast, reduce, DspFloat, Scratch};

/// Samples each partial product adds before the partial products are summed.
pub const COVARIANCE_SPLIT_SAMPLES: usize = 4096;

/// Device buffers of the split products, grown on demand and reused.
#[derive(Debug, Default)]
struct SplitBuffers {
    /// `[splits, channels, COVARIANCE_SPLIT_SAMPLES]` split-major samples.
    splits: Scratch,
    /// `[splits, channels, channels]` partial products.
    partial: Scratch,
}

/// Where the samples come from: columns `cols` of `channels` rows `row_len` apart, centred on
/// `means` when given.
struct Samples<'a> {
    input: &'a Handle,
    channels: usize,
    row_len: usize,
    cols: Range<usize>,
    means: Option<&'a Handle>,
}

/// `out (=|+=) x xᵀ / n` for the `[channels, n]` samples `src`, through split products.
fn gram<F: DspFloat>(client: &Client, src: Samples<'_>, buffers: &mut SplitBuffers, out: &Handle, accumulate: bool) {
    let Samples { input, channels, row_len, cols, means } = src;
    let n = cols.len();
    let splits = n.div_ceil(COVARIANCE_SPLIT_SAMPLES).max(1);
    let (len, split_len) = (channels * channels, channels * COVARIANCE_SPLIT_SAMPLES);
    let total = splits * split_len;
    let split_major = buffers.splits.get::<F>(client, total);
    let partial = buffers.partial.get::<F>(client, splits * len);
    let centred = means.is_some();
    // Uncentred, the kernel never reads `means`: bind `out` (`F`, ≥ `channels` values)
    let means = means.unwrap_or(out).clone();

    let geom = LaunchGeometry::elementwise(client, total);
    // SAFETY: `input` holds `channels · row_len` and `means` ≥ `channels` values of `F`;
    // `split_major` holds `total`
    unsafe {
        split_rows_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), channels * row_len),
            BufferArg::from_raw_parts(means, channels),
            BufferArg::from_raw_parts(split_major.clone(), total),
            cols.start as u32,
            row_len as u32,
            n as u32,
            channels as u32,
            COVARIANCE_SPLIT_SAMPLES as u32,
            total as u32,
            centred,
        );
    }
    let x = MatrixView::batched_row_major(&split_major, total, splits, channels, COVARIANCE_SPLIT_SAMPLES);
    matmul::<F>(client, &x, &x.transposed(), &partial, splits * len);

    let geom = LaunchGeometry::elementwise(client, len);
    // SAFETY: `partial` holds `splits · len` and `out` `len` values of `F`
    unsafe {
        sum_slices_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(partial, splits * len),
            BufferArg::from_raw_parts(out.clone(), len),
            len as u32,
            splits as u32,
            cast::<F>(1.0 / n.max(1) as f64),
            accumulate,
        );
    }
}

/// Covariance (`/ samples`) of a `[channels, samples]` buffer of `F` into `out_cov`
/// (`[channels, channels]`, row-major) and the channel means into `out_mean` (`channels` values).
pub fn covariance<F: DspFloat>(
    client: &Client,
    input: &Handle,
    out_cov: &Handle,
    out_mean: &Handle,
    channels: usize,
    samples: usize,
) {
    if channels == 0 {
        return;
    }
    let std_scratch = buffer::empty::<F>(client, channels);
    reduce::row_mean_std::<F>(client, input, out_mean, &std_scratch, channels, samples);
    let src = Samples { input, channels, row_len: samples, cols: 0..samples, means: Some(out_mean) };
    gram::<F>(client, src, &mut SplitBuffers::default(), out_cov, false);
}

/// Mean over batches of the uncentred second moment `X Xᵀ / samples`, accumulated on the device:
/// every [`Self::add`] adds one batch (columns of a device buffer, e.g. the interior of a
/// halo-padded window) with equal weight, and only [`Self::finish`] downloads the
/// `[channels, channels]` result. This is the whitening covariance of Kilosort4 / EMUsort
/// (`CC += X Xᵀ / n` per batch, `CC / k`), for high-passed data whose mean is ~0.
pub struct SecondMomentAccumulator<F: DspFloat> {
    client: Client,
    channels: usize,
    sum: Handle,
    batches: usize,
    buffers: SplitBuffers,
    _float: std::marker::PhantomData<F>,
}

impl<F: DspFloat> SecondMomentAccumulator<F> {
    pub fn new(client: &Client, channels: usize) -> Self {
        Self {
            client: client.clone(),
            channels,
            sum: buffer::zeros::<F>(client, channels * channels),
            batches: 0,
            buffers: SplitBuffers::default(),
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
        let src = Samples { input, channels: self.channels, row_len, cols, means: None };
        gram::<F>(&self.client, src, &mut self.buffers, &self.sum, true);
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
        buffer::download::<F>(&self.client, self.sum).into_iter().map(|v| crate::core::to_f64(v) / k).collect()
    }
}

/// Covariance and means of host data (`[channels, samples]`, uploaded as `F`): the covariance stays
/// on the device (ready for [`crate::linalg::symmetric_eigen`]), the means come back to the host.
pub fn covariance_of_host<F: DspFloat>(client: &Client, data: &[f32], channels: usize, samples: usize) -> (Handle, Vec<f64>) {
    assert_eq!(data.len(), channels * samples, "data size mismatch");
    let input = buffer::upload(client, &crate::core::cast_f32::<F>(data));
    let cov = buffer::empty::<F>(client, channels * channels);
    let mean = buffer::empty::<F>(client, channels);
    covariance::<F>(client, &input, &cov, &mean, channels, samples);
    let mean = buffer::download::<F>(client, mean).into_iter().map(crate::core::to_f64).collect();
    (cov, mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches_host(client: &Client) {
        for (channels, samples) in [(1usize, 10usize), (5, 9_000), (5, 9_001), (17, 1_234)] {
            let data: Vec<f32> = (0..channels * samples)
                .map(|i| {
                    let (c, t) = ((i / samples) as f32, (i % samples) as f32);
                    (t * 0.013 * (c + 1.0)).sin() * 20.0 + ((i * 7919) % 101) as f32 * 0.1 + c * 50.0
                })
                .collect();
            let input = buffer::upload(client, &data);
            let (cov, mean) = (buffer::empty::<f32>(client, channels * channels), buffer::empty::<f32>(client, channels));
            covariance::<f32>(client, &input, &cov, &mean, channels, samples);
            let cov = buffer::download::<f32>(client, cov);

            let m: Vec<f64> = (0..channels).map(|c| data[c * samples..(c + 1) * samples].iter().map(|v| *v as f64).sum::<f64>() / samples as f64).collect();
            for i in 0..channels {
                for j in 0..channels {
                    let want: f64 = (0..samples)
                        .map(|t| (data[i * samples + t] as f64 - m[i]) * (data[j * samples + t] as f64 - m[j]))
                        .sum::<f64>()
                        / samples as f64;
                    let got = cov[i * channels + j] as f64;
                    assert!((got - want).abs() < 1e-3 * want.abs().max(1.0), "{} {channels}x{samples} [{i},{j}]: {got} vs {want}", client.name());
                }
            }
        }
    }
    runtime_test!(test_covariance_matches_host, matches_host);

    /// Two batches of different lengths, read from the interior columns of padded rows.
    fn second_moment_matches_host(client: &Client) {
        let channels = 3;
        let batches = [(40usize, 5..35usize), (25, 3..20)];
        let mut acc = SecondMomentAccumulator::<f32>::new(client, channels);
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
            assert!((got - want).abs() < 1e-3 * want.abs().max(1.0), "{} [{k}]: {got} vs {want}", client.name());
        }
    }
    runtime_test!(test_second_moment_matches_host, second_moment_matches_host);

    /// On runtimes with `f64`, both paths compute in `f64`: they match the host to ~1e-12, which an
    /// `f32` step anywhere (≈1e-7 relative) would miss.
    fn keeps_f64(client: &Client) {
        if !client.properties().supports_type(f64::elem_type_native()) {
            return;
        }
        let (channels, samples) = (5usize, 9_001usize);
        let data: Vec<f64> = (0..channels * samples).map(|i| ((i * 7919) % 1013) as f64 / 7.0 + (i / samples) as f64 * 1e3).collect();
        let m: Vec<f64> = (0..channels).map(|c| data[c * samples..(c + 1) * samples].iter().sum::<f64>() / samples as f64).collect();
        let input = buffer::upload(client, &data);
        let (cov, mean) = (buffer::empty::<f64>(client, channels * channels), buffer::empty::<f64>(client, channels));
        covariance::<f64>(client, &input, &cov, &mean, channels, samples);
        let cov = buffer::download::<f64>(client, cov);

        let mut acc = SecondMomentAccumulator::<f64>::new(client, channels);
        acc.add(&input, samples, 3..samples);
        let moment = acc.finish();
        for i in 0..channels {
            for j in 0..channels {
                let row = |c: usize| &data[c * samples..(c + 1) * samples];
                let want: f64 = row(i).iter().zip(row(j)).map(|(a, b)| (a - m[i]) * (b - m[j])).sum::<f64>() / samples as f64;
                let got = cov[i * channels + j];
                assert!((got - want).abs() < 1e-12 * want.abs().max(1.0), "{} f64 cov [{i},{j}]: {got} vs {want}", client.name());
                let want: f64 = row(i)[3..].iter().zip(&row(j)[3..]).map(|(a, b)| a * b).sum::<f64>() / (samples - 3) as f64;
                let got = moment[i * channels + j];
                assert!((got - want).abs() < 1e-12 * want.abs().max(1.0), "{} f64 moment [{i},{j}]: {got} vs {want}", client.name());
            }
        }
    }
    runtime_test!(test_covariance_keeps_f64, keeps_f64);
}
