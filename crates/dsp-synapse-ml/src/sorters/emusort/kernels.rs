//! EMUsort channel delays on the device: estimation ([`ChannelDelayEstimator`]) and removal
//! ([`ChannelAligner`]). Batches never leave the device; the estimator downloads its
//! `[channels, channels, 2·max_lag + 1]` cross-correlation once, at the end.
//!
//! Semantics follow EMUsort (checked against `snel-repo/EMUsort` `a06bb60`, `ks4mods`): each
//! channel of a batch is divided by its standard deviation over the padded batch and rectified;
//! `CC[a, b, lag] = mean_t x_a[t − lag] · x_b[t]` over the batch interior, averaged over batches;
//! the reference channel maximizes `Σ_a max_lag CC[a, b, ·]`, and each channel's delay is the lag of
//! its best correlation with the reference. Removal is `x[i, t] ← x[i, (t + delay_i) mod samples]`
//! (a circular shift within the padded batch; with `max_lag` samples of padding only the padding
//! wraps). Lagged reads past the buffer edges repeat the edge sample, as upstream pads its first
//! and last batches.

use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::{buffer, reduce};
use dsp_core::compute::{channel_position, sample_position, LaunchGeometry};

/// `output[ch, t] = input[ch, (t + shift[ch]) mod samples]`, `shift[ch] ≥ 0`.
#[cube(launch)]
pub fn apply_channel_delays_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    shifts: &Array<u32>,
    channels: u32,
    samples: u32,
) {
    let t = sample_position();
    let ch = channel_position();
    if ch < channels && t < samples {
        // shift < samples, so one subtraction wraps
        let mut src_t = t + shifts[ch as usize];
        if src_t >= samples {
            src_t -= samples;
        }
        output[(ch * samples + t) as usize] = input[(ch * samples + src_t) as usize];
    }
}

/// `env[ch, t] = |x[ch, t]| / std[ch]` (`0` for a flat channel).
#[cube(launch)]
pub fn delay_envelope_kernel<F: Float>(input: &Array<F>, std: &Array<F>, env: &mut Array<F>, channels: u32, samples: u32) {
    let t = sample_position();
    let ch = channel_position();
    if ch < channels && t < samples {
        let sd = std[ch as usize];
        let idx = (ch * samples + t) as usize;
        if sd > F::new(0.0f32) {
            env[idx] = F::abs(input[idx]) / sd;
        } else {
            env[idx] = F::new(0.0f32);
        }
    }
}

/// `cc[q] += Σ_t env[a, t − lag] · env[b, t] / samples` over the interior samples
/// `t ∈ [col_start .. col_start + samples)`, for `q = (a · channels + b) · lags + li`,
/// `lag = li − max_lag`; `t − lag` is clamped to the row (only windows whose interior is closer
/// than `max_lag` to an edge take the clamped loop). One unit per `q`, adding to the running sum.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn delay_cc_kernel<F: Float>(
    env: &Array<F>,
    cc: &mut Array<F>,
    channels: u32,
    row_len: u32,
    col_start: u32,
    samples: u32,
    max_lag: u32,
) {
    let lags = 2u32 * max_lag + 1u32;
    let q = ABSOLUTE_POS as u32;
    if q < channels * channels * lags {
        let li = q % lags;
        let pair = q / lags;
        let (base_a, base_b) = ((pair / channels) * row_len, (pair % channels) * row_len);
        // Lagged sample of interior sample t: col_start + t + max_lag − li
        let shift = col_start + max_lag;
        let in_bounds = shift >= li && shift - li + samples <= row_len;
        let mut acc = F::new(0.0f32);
        let mut t = 0u32;
        if in_bounds {
            let start_a = base_a + shift - li;
            let start_b = base_b + col_start;
            while t < samples {
                acc += env[(start_a + t) as usize] * env[(start_b + t) as usize];
                t += 1u32;
            }
        } else {
            while t < samples {
                let shifted = shift + t;
                let mut ta = 0u32;
                if shifted >= li {
                    ta = u32::min(shifted - li, row_len - 1u32);
                }
                acc += env[(base_a + ta) as usize] * env[(base_b + col_start + t) as usize];
                t += 1u32;
            }
        }
        cc[q as usize] += acc / F::cast_from(u32::max(samples, 1u32));
    }
}

/// `(delays, reference)` from a `[channels, channels, 2·max_lag + 1]` mean cross-correlation: the
/// reference channel maximizes `Σ_a max_lag CC[a, b, ·]`; each channel's delay is the lag of its
/// best correlation with the reference. Ties keep the first (smallest lag, lowest channel).
pub fn delays_from_cross_correlation(cc: &[f64], channels: usize, max_lag: usize) -> (Vec<isize>, usize) {
    let lags = 2 * max_lag + 1;
    assert_eq!(cc.len(), channels * channels * lags, "cross-correlation must be [channels, channels, lags]");
    if channels == 0 {
        return (Vec::new(), 0);
    }
    let peak = |a: usize, b: usize| {
        let row = &cc[(a * channels + b) * lags..(a * channels + b + 1) * lags];
        row.iter().enumerate().fold((0usize, f64::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best })
    };
    let reference = (0..channels)
        .map(|b| (b, (0..channels).map(|a| peak(a, b).1).sum::<f64>()))
        .fold((0usize, f64::NEG_INFINITY), |best, x| if x.1 > best.1 { x } else { best })
        .0;
    let delays = (0..channels).map(|b| peak(reference, b).0 as isize - max_lag as isize).collect();
    (delays, reference)
}

/// Accumulates the lagged cross-correlation of channel envelopes over batches on the device. Its
/// scratch buffers are kept between batches; nothing is read back until the end.
pub struct ChannelDelayEstimator<R: Runtime> {
    client: ComputeClient<R>,
    channels: usize,
    max_lag: usize,
    /// `[channels, channels, 2·max_lag + 1]` sum of per-batch means.
    cc: Handle,
    batches: usize,
    /// Per-channel mean and standard deviation of the current batch.
    mean: Handle,
    std: Handle,
    /// `[channels, capacity]` envelope of the current batch.
    env: Handle,
    capacity: usize,
}

impl<R: Runtime> ChannelDelayEstimator<R> {
    pub fn new(client: &ComputeClient<R>, channels: usize, max_lag: usize) -> Self {
        Self {
            client: client.clone(),
            channels,
            max_lag,
            cc: buffer::zeros::<R, f32>(client, channels * channels * (2 * max_lag + 1)),
            batches: 0,
            mean: buffer::empty::<R, f32>(client, channels),
            std: buffer::empty::<R, f32>(client, channels),
            env: buffer::empty::<R, f32>(client, 1),
            capacity: 0,
        }
    }

    /// Adds a `[channels, row_len]` device batch (`f32`, padded): the standard deviation is taken
    /// over the whole row, the correlation over the interior `cols`. Empty interiors are ignored.
    pub fn add(&mut self, input: &Handle, row_len: usize, cols: Range<usize>) {
        assert!(cols.end <= row_len, "interior {cols:?} outside rows of {row_len}");
        let channels = self.channels;
        if channels == 0 || cols.is_empty() {
            return;
        }
        if row_len > self.capacity {
            self.env = buffer::empty::<R, f32>(&self.client, channels * row_len);
            self.capacity = row_len;
        }
        let client = &self.client;
        reduce::row_mean_std::<R, f32>(client, input, &self.mean, &self.std, channels, row_len);

        let total = channels * row_len;
        let geom = LaunchGeometry::channels_samples(client, channels, row_len);
        unsafe {
            delay_envelope_kernel::launch::<f32, R>(
                client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(input.clone(), total),
                ArrayArg::from_raw_parts(self.std.clone(), channels),
                ArrayArg::from_raw_parts(self.env.clone(), total),
                channels as u32,
                row_len as u32,
            );
        }

        let triples = channels * channels * (2 * self.max_lag + 1);
        let geom = LaunchGeometry::elementwise(client, triples);
        unsafe {
            delay_cc_kernel::launch::<f32, R>(
                client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(self.env.clone(), total),
                ArrayArg::from_raw_parts(self.cc.clone(), triples),
                channels as u32,
                row_len as u32,
                cols.start as u32,
                cols.len() as u32,
                self.max_lag as u32,
            );
        }
        self.batches += 1;
    }

    /// Batches added so far.
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// The mean cross-correlation over batches, `[channels, channels, 2·max_lag + 1]` (one
    /// download); zeros before any batch.
    pub fn cross_correlation(self) -> Vec<f64> {
        let k = self.batches.max(1) as f64;
        buffer::download::<R, f32>(&self.client, self.cc).into_iter().map(|v| v as f64 / k).collect()
    }

    /// `(delays, reference)` ([`delays_from_cross_correlation`]); zeros and channel 0 before any
    /// batch.
    pub fn delays(self) -> (Vec<isize>, usize) {
        let (channels, max_lag, batches) = (self.channels, self.max_lag, self.batches);
        if batches == 0 {
            return (vec![0; channels], 0);
        }
        delays_from_cross_correlation(&self.cross_correlation(), channels, max_lag)
    }
}

/// Removes per-channel delays from device batches into one persistent output buffer. The shifts
/// are uploaded again only when the batch length changes (in a schedule: the first and last
/// windows).
pub struct ChannelAligner<R: Runtime> {
    client: ComputeClient<R>,
    delays: Vec<isize>,
    max_samples: usize,
    output: Handle,
    /// `(samples, shifts)` of the last batch length.
    shifts: Option<(usize, Handle)>,
}

impl<R: Runtime> ChannelAligner<R> {
    /// Aligner for batches of at most `max_samples` samples per channel.
    pub fn new(client: &ComputeClient<R>, delays: Vec<isize>, max_samples: usize) -> Self {
        let output = buffer::empty::<R, f32>(client, delays.len() * max_samples);
        Self { client: client.clone(), delays, max_samples, output, shifts: None }
    }

    pub fn delays(&self) -> &[isize] {
        &self.delays
    }

    /// The aligned `[channels, samples]` batch of a `[channels, samples]` device batch (`f32`).
    /// The returned handle is the aligner's buffer: valid until the next call.
    pub fn align(&mut self, input: &Handle, samples: usize) -> Handle {
        assert!(samples <= self.max_samples, "batch of {samples} samples exceeds the aligner's {}", self.max_samples);
        let channels = self.delays.len();
        let shifts = match &self.shifts {
            Some((n, handle)) if *n == samples => handle.clone(),
            _ => {
                let host: Vec<u32> = self.delays.iter().map(|&d| d.rem_euclid(samples.max(1) as isize) as u32).collect();
                let handle = buffer::upload(&self.client, &host);
                self.shifts = Some((samples, handle.clone()));
                handle
            }
        };
        let total = channels * samples;
        let geom = LaunchGeometry::channels_samples(&self.client, channels, samples);
        unsafe {
            apply_channel_delays_kernel::launch::<f32, R>(
                &self.client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(input.clone(), total),
                ArrayArg::from_raw_parts(self.output.clone(), total),
                ArrayArg::from_raw_parts(shifts, channels),
                channels as u32,
                samples as u32,
            );
        }
        self.output.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::{ComputeTarget, ComputeTask};

    /// The reference semantics, on the host, as an oracle for the device path.
    fn host_cross_correlation(x: &[f32], channels: usize, row_len: usize, cols: Range<usize>, max_lag: usize) -> Vec<f64> {
        let env: Vec<f64> = x
            .chunks_exact(row_len)
            .flat_map(|row| {
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / row_len as f64;
                let sd = (row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / row_len as f64).sqrt();
                row.iter().map(move |&v| if sd > 0.0 { (v as f64).abs() / sd } else { 0.0 }).collect::<Vec<_>>()
            })
            .collect();
        let lags = 2 * max_lag + 1;
        let mut cc = vec![0.0; channels * channels * lags];
        for a in 0..channels {
            for b in 0..channels {
                for li in 0..lags {
                    let s: f64 = cols
                        .clone()
                        .map(|t| {
                            let ta = (t as isize - (li as isize - max_lag as isize)).clamp(0, row_len as isize - 1) as usize;
                            env[a * row_len + ta] * env[b * row_len + t]
                        })
                        .sum();
                    cc[(a * channels + b) * lags + li] = s / cols.len() as f64;
                }
            }
        }
        cc
    }

    fn host_align(x: &[f32], samples: usize, delays: &[isize]) -> Vec<f32> {
        let mut out = x.to_vec();
        for (row, &d) in out.chunks_exact_mut(samples).zip(delays) {
            row.rotate_left(d.rem_euclid(samples as isize) as usize);
        }
        out
    }

    /// A spike train on channel 0, copied to channel `c` `true_delays[c]` samples later.
    fn delayed_batch(channels: usize, samples: usize, true_delays: &[usize]) -> Vec<f32> {
        let base: Vec<f32> = (0..samples).map(|t| if t % 37 == 5 { 10.0 } else { ((t * 13) % 7) as f32 * 0.1 }).collect();
        (0..channels).flat_map(|c| (0..samples).map(|t| base[(t + samples - true_delays[c]) % samples]).collect::<Vec<_>>()).collect()
    }

    struct Estimate(Vec<f32>, usize, usize, Range<usize>, usize);
    impl ComputeTask for Estimate {
        type Output = Vec<f64>;
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
            let Estimate(x, channels, row_len, cols, max_lag) = self;
            let mut est = ChannelDelayEstimator::new(&client, channels, max_lag);
            est.add(&buffer::upload(&client, &x), row_len, cols);
            est.cross_correlation()
        }
    }

    #[test]
    fn device_cross_correlation_matches_host_and_finds_delays() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let (channels, row_len, max_lag) = (3usize, 400usize, 8usize);
        let x = delayed_batch(channels, row_len, &[0, 3, 6]);
        // Interior touching the left edge exercises the clamped reads
        for cols in [20..380, 0..380] {
            let got = target.run(Estimate(x.clone(), channels, row_len, cols.clone(), max_lag)).expect("target run");
            let want = host_cross_correlation(&x, channels, row_len, cols, max_lag);
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-3 * w.abs().max(1.0), "[{k}]: {g} vs {w}");
            }
        }
        let cc = host_cross_correlation(&x, channels, row_len, 20..380, max_lag);
        let (delays, reference) = delays_from_cross_correlation(&cc, channels, max_lag);
        assert_eq!(delays[reference], 0);
        let aligned = host_align(&x, row_len, &delays);
        for c in 1..channels {
            assert_eq!(&aligned[c * row_len + 20..c * row_len + 380], &aligned[20..380], "channel {c} not aligned (delays {delays:?})");
        }
    }

    struct Align(Vec<f32>, Vec<isize>, Vec<usize>);
    impl ComputeTask for Align {
        type Output = Vec<Vec<f32>>;
        fn run<R: Runtime>(self, client: ComputeClient<R>) -> Self::Output {
            let Align(x, delays, lengths) = self;
            let channels = delays.len();
            let mut aligner = ChannelAligner::new(&client, delays, *lengths.iter().max().unwrap());
            lengths
                .into_iter()
                .map(|n| {
                    let out = aligner.align(&buffer::upload(&client, &x[..channels * n]), n);
                    buffer::download::<R, f32>(&client, out)[..channels * n].to_vec()
                })
                .collect()
        }
    }

    #[test]
    fn device_alignment_matches_host_across_batch_lengths() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let delays = vec![0isize, 5, -3];
        let x: Vec<f32> = (0..3 * 100).map(|v| v as f32).collect();
        let lengths = vec![100usize, 100, 60, 100];
        let got = target.run(Align(x.clone(), delays.clone(), lengths.clone())).expect("target run");
        for (out, n) in got.iter().zip(lengths) {
            assert_eq!(out, &host_align(&x[..3 * n], n, &delays), "length {n}");
        }
    }
}
