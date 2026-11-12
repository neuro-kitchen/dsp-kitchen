//! Kilosort4's drift correction (paper, Methods, *Drift correction*; behaviour matched to upstream's
//! `datashift.py` and `get_drift_matrix` by reading them; no code copied).
//!
//! **Estimate** ([`estimate_drift`], on the host: the spikes are there already):
//! 1. A *fingerprint* per batch: the batch's universal-template spikes counted on a grid of depth
//!    (`binning_depth_um` bins from 1 µm below the lowest channel) × 20 amplitude bins
//!    (logarithmic from the detection threshold to 100), as `log2(1 + count)`.
//! 2. **Rigid alignment:** the fingerprints, mean-subtracted along depth, are shifted (circularly, by
//!    −15…15 bins) to best match a target (first the middle batch, or batch 300); the target
//!    becomes the mean of the aligned fingerprints; ten rounds, the shifts accumulating.
//! 3. **Fine alignment** in `2·nblocks − 1` half-overlapping depth blocks (one for rigid
//!    correction): each block's match over −5…5 bins, smoothed by a Gaussian over (shift, batch,
//!    block) (`smoothing`; scipy's `gaussian_filter`: reflect edges, radius `⌊4σ + ½⌋`),
//!    upsampled ×10 by a Gaussian kernel and its best shift added.
//! 4. `dshift[batch, block]` in µm.
//!
//! **Correction** ([`DriftCorrector`], on the device, a step after whitening): each window `X`
//! becomes `M · X` with `M = K(y − shift, x) · (K(x, x) + 0.01·I)⁻¹`, `K` a Gaussian kernel over
//! the channel positions (`sig_interp_um`): kriging interpolation of the signal at the depths the
//! channels had before the drift. With several blocks, each channel's shift is interpolated
//! linearly between the blocks' centres (extrapolated past the ends).
//!
//! **What crosses the bus:** the inverse kernel and the channel positions go up once; per window,
//! one shift per channel (`channels` floats). `M` is built and applied on the device.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dspk_core::compute::LaunchGeometry;
use dspk_core::HaloWindow;
use dspk_ops::core::buffer;
use dspk_ops::linalg::{matmul, MatrixView};

pub mod kernels;
use kernels::kriging_kernel;

/// Amplitude bins of a fingerprint (upstream: always 20).
const AMP_BINS: usize = 20;
/// Coarse search: shifts of −15…15 depth bins, ten rounds.
const COARSE_SHIFTS: i32 = 15;
const ROUNDS: usize = 10;
/// Fine search in blocks: −5…5 bins, upsampled ×10 (101 points).
const FINE_SHIFTS: i32 = 5;
const UPSAMPLE: usize = 10;
/// The target's first batch: 300 if the recording has that many, else the middle one.
const TARGET_BATCH: usize = 300;
/// Ridge on the kriging kernel before inversion (upstream: 0.01).
const KRIGING_RIDGE: f64 = 0.01;
/// Largest amplitude binned (upstream: 99, on a scale to 100).
const AMP_MAX: f32 = 99.0;

/// Drift-correction settings (upstream defaults, `kilosort/parameters.py`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftOptions {
    /// Non-overlapping depth blocks; `nblocks − 1` half-overlapping ones are added between them.
    /// 1: rigid correction (Kilosort4's default); 0: no drift correction (EMUsort's default).
    pub nblocks: usize,
    /// Depth bin of the fingerprints (µm).
    pub binning_depth_um: f32,
    /// Spatial smoothness of the kriging interpolation (µm).
    pub sig_interp_um: f32,
    /// Gaussian σ of the block-match smoothing along (shift, batch, block); the block axis has no
    /// effect with one block.
    pub smoothing: [f32; 3],
}

impl Default for DriftOptions {
    fn default() -> Self {
        Self { nblocks: 1, binning_depth_um: 5.0, sig_interp_um: 20.0, smoothing: [0.5, 0.5, 0.5] }
    }
}

impl DriftOptions {
    /// No drift correction.
    pub fn off() -> Self {
        Self { nblocks: 0, ..Self::default() }
    }

    pub fn enabled(&self) -> bool {
        self.nblocks > 0
    }
}

/// One spike of the drift pass: its batch, depth (µm) and amplitude (whitened σ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftSpike {
    pub batch: usize,
    pub y_um: f32,
    pub amplitude: f32,
}

/// The estimated drift: `dshift[batch · blocks + block]` (µm) and each block's centre depth.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftModel {
    pub batches: usize,
    pub blocks: usize,
    pub dshift: Vec<f32>,
    pub block_centres_um: Vec<f32>,
}

impl DriftModel {
    /// As the sorting's drift track: one value per batch (µm; the mean over blocks, the shift the
    /// correction applied), at the batch's centre time.
    pub fn estimate(&self, total_samples: u64, sample_rate_hz: f64) -> dspk_synapse::spatial::drift::DriftEstimate {
        let batch_sec = total_samples as f64 / sample_rate_hz / self.batches.max(1) as f64;
        dspk_synapse::spatial::drift::DriftEstimate {
            time_bin_centers_sec: (0..self.batches).map(|b| (b as f64 + 0.5) * batch_sec).collect(),
            drift_um: (0..self.batches).map(|b| self.dshift[b * self.blocks..(b + 1) * self.blocks].iter().sum::<f32>() / self.blocks as f32).collect(),
            activity_map: Vec::new(),
            num_time_bins: self.batches,
            num_depth_bins: 0,
            depth_min_um: 0.0,
            depth_bin_size_um: 0.0,
        }
    }

    /// Shift of every channel at depths `channel_y` in `batch` (µm): the batch's shift (one block),
    /// else interpolated linearly between the block centres, extrapolated past the ends.
    pub fn channel_shifts(&self, batch: usize, channel_y: &[f32]) -> Vec<f32> {
        let b = batch.min(self.batches.saturating_sub(1));
        let row = &self.dshift[b * self.blocks..(b + 1) * self.blocks];
        if self.blocks == 1 {
            return vec![row[0]; channel_y.len()];
        }
        let xs = &self.block_centres_um;
        channel_y
            .iter()
            .map(|&y| {
                // The segment holding y (the first or last for extrapolation)
                let mut k = 0;
                while k + 2 < xs.len() && y > xs[k + 1] {
                    k += 1;
                }
                let (x0, x1, y0, y1) = (xs[k], xs[k + 1], row[k], row[k + 1]);
                if x1 == x0 { y0 } else { y0 + (y - x0) * (y1 - y0) / (x1 - x0) }
            })
            .collect()
    }
}

/// The fingerprints `[batches, depth_bins, AMP_BINS]` (`log2(1 + count)`) and the depth bins'
/// centres (µm).
pub fn fingerprints(spikes: &[DriftSpike], batches: usize, channel_y: &[f32], th_universal: f32, binning_depth_um: f32) -> (Vec<f64>, Vec<f64>, usize) {
    let ymin = channel_y.iter().copied().fold(f32::INFINITY, f32::min) as f64;
    let ymax = channel_y.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let dd = binning_depth_um as f64;
    let dmin = ymin - 1.0;
    let bins = 1 + ((ymax - dmin) / dd).ceil() as usize;
    let mut f = vec![0.0f64; batches * bins * AMP_BINS];
    let (lo, span) = ((th_universal as f64).log10(), 100f64.log10() - (th_universal as f64).log10());
    for s in spikes {
        if s.batch >= batches {
            continue;
        }
        let row = ((s.y_um as f64 - dmin) / dd) as i64;
        let amp = ((s.amplitude.min(AMP_MAX) as f64).log10() - lo) / span;
        let col = (1e-5 + amp * AMP_BINS as f64) as i64;
        if row < 0 || row >= bins as i64 || col < 0 || col >= AMP_BINS as i64 {
            continue;
        }
        f[(s.batch * bins + row as usize) * AMP_BINS + col as usize] += 1.0;
    }
    f.iter_mut().for_each(|v| *v = (1.0 + *v).log2());
    let centres = (0..bins).map(|k| dmin + dd * k as f64 - dd / 2.0).collect();
    (f, centres, bins)
}

/// `x` rolled by `s` along its `n` rows (`out[i] = x[(i − s) mod n]`, rows of `w` values).
fn roll_rows(x: &[f64], n: usize, w: usize, s: i32) -> Vec<f64> {
    let mut out = vec![0.0; n * w];
    for i in 0..n {
        let src = (i as i64 - s as i64).rem_euclid(n as i64) as usize;
        out[i * w..(i + 1) * w].copy_from_slice(&x[src * w..(src + 1) * w]);
    }
    out
}

/// Mean of `a · b` over equal-length slices.
fn mean_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / a.len().max(1) as f64
}

/// First index of the largest value.
fn argmax(v: impl Iterator<Item = f64>) -> usize {
    let mut best = (0usize, f64::NEG_INFINITY);
    for (i, x) in v.enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    best.0
}

/// `n` points from `a` to `b` (numpy's `linspace`).
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    (0..n).map(|k| a + (b - a) * k as f64 / (n - 1) as f64).collect()
}

/// scipy's `gaussian_filter1d` weights for `sigma` (radius `⌊4σ + ½⌋`, normalized).
fn gaussian_weights(sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma + 0.5) as i64;
    let w: Vec<f64> = (-radius..=radius).map(|x| (-0.5 * (x * x) as f64 / (sigma * sigma)).exp()).collect();
    let total: f64 = w.iter().sum();
    w.into_iter().map(|v| v / total).collect()
}

/// scipy's `reflect` edge: `d c b a | a b c d | d c b a`.
fn reflect(i: i64, n: usize) -> usize {
    let n = n as i64;
    let period = 2 * n;
    let mut j = i.rem_euclid(period);
    if j >= n {
        j = period - 1 - j;
    }
    j as usize
}

/// A Gaussian along each axis of a 3-D array `[a0, a1, a2]` (scipy's `gaussian_filter`, mode
/// reflect); an axis with σ ≈ 0 is left alone.
pub fn gaussian_filter_3d(x: &mut [f64], dims: [usize; 3], sigma: [f64; 3]) {
    let stride = [dims[1] * dims[2], dims[2], 1];
    for axis in 0..3 {
        if sigma[axis] <= 1e-15 || dims[axis] == 0 {
            continue;
        }
        let w = gaussian_weights(sigma[axis]);
        let r = (w.len() / 2) as i64;
        let n = dims[axis];
        let src = x.to_vec();
        for i0 in 0..dims[0] {
            for i1 in 0..dims[1] {
                for i2 in 0..dims[2] {
                    let idx = [i0, i1, i2];
                    let base = i0 * stride[0] + i1 * stride[1] + i2 * stride[2] - idx[axis] * stride[axis];
                    let mut acc = 0.0;
                    for (k, wk) in w.iter().enumerate() {
                        let j = reflect(idx[axis] as i64 + k as i64 - r, n);
                        acc += wk * src[base + j * stride[axis]];
                    }
                    x[i0 * stride[0] + i1 * stride[1] + i2 * stride[2]] = acc;
                }
            }
        }
    }
}

/// Drift of a recording from its universal-template spikes (module docs). `channel_y`: the
/// channels' depths (µm); `th_universal`: the detection threshold the amplitudes start at.
pub fn estimate_drift(spikes: &[DriftSpike], batches: usize, channel_y: &[f32], th_universal: f32, opts: &DriftOptions) -> DriftModel {
    assert!(opts.enabled(), "drift correction is off (nblocks = 0)");
    let (f, ysamp, nd) = fingerprints(spikes, batches, channel_y, th_universal, opts.binning_depth_um);
    let na = AMP_BINS;
    let size = nd * na;
    // Mean-subtracted along depth, per batch and amplitude bin
    let mut fg = f.clone();
    for b in 0..batches {
        for a in 0..na {
            let mean = (0..nd).map(|d| fg[(b * nd + d) * na + a]).sum::<f64>() / nd as f64;
            (0..nd).for_each(|d| fg[(b * nd + d) * na + a] -= mean);
        }
    }
    let mut target = fg[TARGET_BATCH.min(batches / 2) * size..][..size].to_vec();
    let coarse: Vec<i32> = (-COARSE_SHIFTS..=COARSE_SHIFTS).collect();
    let mut dall = vec![vec![0.0f64; batches]; ROUNDS];
    for round in 0..ROUNDS {
        if round + 1 < ROUNDS {
            for b in 0..batches {
                let fb = fg[b * size..(b + 1) * size].to_vec();
                let best = argmax(coarse.iter().map(|&s| mean_product(&roll_rows(&fb, nd, na, s), &target)));
                fg[b * size..(b + 1) * size].copy_from_slice(&roll_rows(&fb, nd, na, coarse[best]));
                dall[round][b] = coarse[best] as f64;
            }
        }
        // The new target: the mean of the aligned fingerprints
        target = (0..size).map(|i| (0..batches).map(|b| fg[b * size + i]).sum::<f64>() / batches.max(1) as f64).collect();
    }

    // Blocks: `nblocks` along depth plus the half-overlapping ones between them
    let yl = nd / opts.nblocks;
    let firsts: Vec<usize> = linspace(0.0, (nd - yl) as f64, 2 * opts.nblocks - 1).into_iter().map(|v| v.round_ties_even() as usize).collect();
    let blocks = firsts.len();
    let fine: Vec<i32> = (-FINE_SHIFTS..=FINE_SHIFTS).collect();
    let nf = fine.len();
    let mut dcs = vec![0.0f64; nf * batches * blocks];
    let mut centres = vec![0.0f32; blocks];
    for (j, &first) in firsts.iter().enumerate() {
        centres[j] = (ysamp[first..first + yl].iter().sum::<f64>() / yl as f64) as f32;
        let sub_target = &target[first * na..(first + yl) * na];
        for b in 0..batches {
            let sub = &fg[b * size + first * na..b * size + (first + yl) * na];
            for (t, &s) in fine.iter().enumerate() {
                dcs[(t * batches + b) * blocks + j] = mean_product(&roll_rows(sub, yl, na, s), sub_target);
            }
        }
    }
    gaussian_filter_3d(&mut dcs, [nf, batches, blocks], opts.smoothing.map(|s| s as f64));
    let up = linspace(-FINE_SHIFTS as f64, FINE_SHIFTS as f64, 2 * FINE_SHIFTS as usize * UPSAMPLE + 1);
    // Kn[t][u] = exp(−(fine[t] − up[u])² / 2)
    let kn = |t: usize, u: usize| (-((fine[t] as f64 - up[u]).powi(2)) / 2.0).exp();
    let coarse_total: Vec<f64> = (0..batches).map(|b| dall[..ROUNDS - 1].iter().map(|r| r[b]).sum()).collect();
    let mut dshift = vec![0.0f32; batches * blocks];
    for j in 0..blocks {
        for b in 0..batches {
            let best = argmax((0..up.len()).map(|u| (0..nf).map(|t| kn(t, u) * dcs[(t * batches + b) * blocks + j]).sum::<f64>()));
            dshift[b * blocks + j] = ((coarse_total[b] + up[best]) * opts.binning_depth_um as f64) as f32;
        }
    }
    DriftModel { batches, blocks, dshift, block_centres_um: centres }
}

/// `(K + ridge·I)⁻¹` of the Gaussian kernel over `positions` (`[x, y]` µm), row-major, by
/// Gauss-Jordan elimination with partial pivoting in `f64`.
pub fn inverse_kernel(positions: &[[f32; 2]], sigma_um: f32) -> Vec<f64> {
    let n = positions.len();
    let s2 = 2.0 * (sigma_um as f64).powi(2);
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let (dx, dy) = ((positions[i][0] - positions[j][0]) as f64, (positions[i][1] - positions[j][1]) as f64);
            a[i * n + j] = (-(dx * dx + dy * dy) / s2).exp() + if i == j { KRIGING_RIDGE } else { 0.0 };
        }
    }
    let mut inv = vec![0.0f64; n * n];
    (0..n).for_each(|i| inv[i * n + i] = 1.0);
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| a[x * n + c].abs().total_cmp(&a[y * n + c].abs())).unwrap_or(c);
        if p != c {
            for k in 0..n {
                a.swap(c * n + k, p * n + k);
                inv.swap(c * n + k, p * n + k);
            }
        }
        let d = a[c * n + c];
        for k in 0..n {
            a[c * n + k] /= d;
            inv[c * n + k] /= d;
        }
        for r in 0..n {
            if r != c {
                let factor = a[r * n + c];
                if factor != 0.0 {
                    for k in 0..n {
                        a[r * n + k] -= factor * a[c * n + k];
                        inv[r * n + k] -= factor * inv[c * n + k];
                    }
                }
            }
        }
    }
    inv
}

/// Drift correction as a per-window step after whitening (module docs): the window `X` becomes
/// `K(y − shift, x) · (K(x, x) + 0.01·I)⁻¹ · X`, its batch's shifts.
pub struct DriftCorrector {
    client: Client,
    model: DriftModel,
    channel_y: Vec<f32>,
    channels: usize,
    max_samples: usize,
    sigma_um: f32,
    x: Handle,
    y: Handle,
    inverse: Handle,
    kyx: Handle,
    m: Handle,
    output: Handle,
}

impl DriftCorrector {
    /// For windows of at most `max_samples` samples of channels at `positions` (`[x, y]` µm).
    pub fn new(client: &Client, model: DriftModel, positions: &[[f32; 2]], sigma_um: f32, max_samples: usize) -> Self {
        let channels = positions.len();
        let inverse: Vec<f32> = inverse_kernel(positions, sigma_um).into_iter().map(|v| v as f32).collect();
        let xs: Vec<f32> = positions.iter().map(|p| p[0]).collect();
        let channel_y: Vec<f32> = positions.iter().map(|p| p[1]).collect();
        Self {
            client: client.clone(),
            model,
            channels,
            max_samples,
            sigma_um,
            x: buffer::upload(client, &xs),
            y: buffer::upload(client, &channel_y),
            inverse: buffer::upload(client, &inverse),
            kyx: buffer::empty::<f32>(client, channels * channels),
            m: buffer::empty::<f32>(client, channels * channels),
            output: buffer::empty::<f32>(client, channels * max_samples),
            channel_y,
        }
    }

    pub fn model(&self) -> &DriftModel {
        &self.model
    }

    /// The corrected `[channels, samples]` window of batch `batch` (the corrector's buffer: valid
    /// until the next call).
    pub fn correct(&mut self, input: &Handle, batch: usize, samples: usize) -> Handle {
        assert!(samples <= self.max_samples, "window of {samples} samples exceeds the corrector's {}", self.max_samples);
        let c = self.channels;
        let shifts = buffer::upload(&self.client, &self.model.channel_shifts(batch, &self.channel_y));
        let geom = LaunchGeometry::elementwise(&self.client, c * c);
        // SAFETY: `x`, `y`, `shifts` hold `c`, `kyx` `c²` values
        unsafe {
            kriging_kernel::launch::<f32>(
                &self.client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(self.x.clone(), c),
                BufferArg::from_raw_parts(self.y.clone(), c),
                BufferArg::from_raw_parts(shifts, c),
                BufferArg::from_raw_parts(self.kyx.clone(), c * c),
                c as u32,
                1.0 / (2.0 * self.sigma_um * self.sigma_um),
            );
        }
        let kyx = MatrixView::row_major(&self.kyx, c * c, c, c);
        let inverse = MatrixView::row_major(&self.inverse, c * c, c, c);
        matmul::<f32>(&self.client, &kyx, &inverse, &self.m, c * c);
        let m = MatrixView::row_major(&self.m, c * c, c, c);
        let x = MatrixView::row_major(input, c * samples, c, samples);
        matmul::<f32>(&self.client, &m, &x, &self.output, c * self.max_samples);
        self.output.clone()
    }
}

/// Drift correction as [`DeviceWindows`](dspk_synapse::sorter::DeviceWindows)' per-window step:
/// the window's index in the schedule is its batch.
impl dspk_synapse::sorter::WindowAligner for DriftCorrector {
    fn align(&mut self, window: &Handle, at: &HaloWindow, samples: usize) -> Handle {
        self.correct(window, at.index, samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// scipy's `gaussian_filter1d([1, 2, 3, 4, 5], 0.5)` (mode reflect), values from scipy.
    #[test]
    fn gaussian_matches_scipy() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        gaussian_filter_3d(&mut x, [5, 1, 1], [0.5, 0.0, 0.0]);
        let want = [1.10724237, 2.00026387, 3.0, 3.99973613, 4.89275763];
        for (g, w) in x.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{x:?}");
        }
        // One element along an axis: unchanged
        let mut one = vec![3.0];
        gaussian_filter_3d(&mut one, [1, 1, 1], [0.5, 0.5, 0.5]);
        assert!((one[0] - 3.0).abs() < 1e-12, "{one:?}");
    }

    /// A probe drifting by whole depth bins: the estimate follows it (rigid, sign as upstream:
    /// the shift that brings a batch back onto the target).
    #[test]
    fn rigid_drift_is_recovered() {
        let channel_y: Vec<f32> = (0..96).map(|i| (i / 2) as f32 * 20.0).collect();
        let batches = 12;
        // Three "units" at fixed depths, the whole probe moving 0, 5, 10 … µm per batch
        let mut spikes = Vec::new();
        for b in 0..batches {
            let drift = 5.0 * b as f32;
            for (depth, amp) in [(200.0f32, 20.0f32), (420.0, 40.0), (700.0, 15.0)] {
                for k in 0..40 {
                    spikes.push(DriftSpike { batch: b, y_um: depth + drift + (k % 3) as f32, amplitude: amp });
                }
            }
        }
        let model = estimate_drift(&spikes, batches, &channel_y, 10.0, &DriftOptions::default());
        assert_eq!((model.batches, model.blocks), (12, 1));
        // Relative to batch 0, each batch's shift steps by one bin (5 µm)
        let d: Vec<f32> = model.dshift.iter().map(|&s| s - model.dshift[0]).collect();
        for (b, v) in d.iter().enumerate() {
            assert!((v + 5.0 * b as f32).abs() <= 1.0, "batch {b}: {v} (all {d:?})");
        }
    }

    /// `(K + 0.01·I)⁻¹ · (K + 0.01·I) = I`.
    #[test]
    fn inverse_kernel_inverts() {
        let positions: Vec<[f32; 2]> = (0..20).map(|i| [((i % 2) * 32) as f32, (i / 2) as f32 * 20.0]).collect();
        let inv = inverse_kernel(&positions, 20.0);
        let n = positions.len();
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    let (dx, dy) = ((positions[k][0] - positions[j][0]) as f64, (positions[k][1] - positions[j][1]) as f64);
                    let a = (-(dx * dx + dy * dy) / 800.0).exp() + if k == j { 0.01 } else { 0.0 };
                    acc += inv[i * n + k] * a;
                }
                assert!((acc - if i == j { 1.0 } else { 0.0 }).abs() < 1e-8, "({i}, {j}): {acc}");
            }
        }
    }

    /// Several blocks: shifts interpolated between block centres, extrapolated past the ends.
    #[test]
    fn channel_shifts_interpolate_between_blocks() {
        let model = DriftModel { batches: 1, blocks: 3, dshift: vec![0.0, 10.0, 30.0], block_centres_um: vec![100.0, 200.0, 300.0] };
        assert_eq!(model.channel_shifts(0, &[50.0, 150.0, 250.0, 400.0]), vec![-5.0, 5.0, 20.0, 50.0]);
    }
}
