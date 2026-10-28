//! SpikeInterface's `matched_filtering` peak detection (`sortingcomponents/peak_detection/
//! matched_filtering.py`, MIT), the detection of SpyKING CIRCUS 2, on the device.
//!
//! 1. Every channel is correlated with the normalised **prototype** waveform (the valid part;
//!    correlation index `i` covers samples `i .. i + len`, its peak sample is `i + nbefore`).
//! 2. A **spatial sum** per putative source: row `(z, t)` adds the correlations of the channels
//!    near channel `t` with the `exponential_3d` weights of a source at depth `z`
//!    ([`convolution_weights`]). With `peak_sign` both, the templates are doubled with negated
//!    weights.
//! 3. **Thresholds**: `detect_threshold` × each row's noise, the median absolute deviation (around
//!    the median, `/ 0.6745`) of the rows on random chunks of the recording ([`MatchedFilter::fit_thresholds`]).
//! 4. **Peaks**: local maxima of each row reaching its threshold (the device peak finder), then
//!    [`super::locally_exclusive`] over templates whose channels are within `radius_um`, scored by
//!    value / threshold, ties to the earlier sample then the smaller depth index.
//!
//! The prototype's peak is negative for `peak_sign` neg (the correlation of a spike with it is
//! then positive); upstream flips the sign so every detection is a positive peak, as here.
//! Choice of ours: the thresholds' random chunks are correlated one by one (upstream concatenates
//! them first, which adds a junction artefact at each join).

use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::peaks::{find_peak_candidates, Polarity};
use dsp_core::compute::LaunchGeometry;

use super::detect::PeakSign;
use super::exclusive::{locally_exclusive, Candidate};
use super::kernels::{correlate_prototype_kernel, gather_points_kernel, spatial_sum_kernel};

/// `0.6744897501960817`: the MAD of a standard normal.
const MAD_TO_SIGMA: f64 = 0.674_489_750_196_081_7;

/// Settings of [`MatchedFilter`] (upstream defaults in [`Default`]).
#[derive(Debug, Clone, PartialEq)]
pub struct MatchedFilterOptions {
    pub detect_threshold: f64,
    /// Samples.
    pub exclude_sweep: usize,
    pub radius_um: f32,
    pub peak_sign: PeakSign,
    /// `get_convolution_weights`: source depths (µm) and decay (µm).
    pub z_list_um: Vec<f64>,
    pub sigma_3d: f64,
    /// Weights below this are dropped; `None`: `0.5 / √channels`.
    pub sparsity_threshold: Option<f64>,
}

impl Default for MatchedFilterOptions {
    fn default() -> Self {
        Self {
            detect_threshold: 5.0,
            exclude_sweep: 30,
            radius_um: 50.0,
            peak_sign: PeakSign::Neg,
            z_list_um: (0..5).map(|i| 30.0 * i as f64).collect(),
            sigma_3d: 2.5,
            sparsity_threshold: None,
        }
    }
}

/// Upstream `get_convolution_weights(mode="exponential_3d")`: `[z, i, j]` = `exp(−√(d_ij² + z²) /
/// σ)`, each column `j` scaled to unit norm over `i`, entries below the sparsity threshold set to
/// 0, columns normalised again (all-zero columns stay 0). Row-major `[z_list.len(), n, n]`.
pub fn convolution_weights(positions: &[[f32; 2]], z_list_um: &[f64], sigma_3d: f64, sparsity_threshold: Option<f64>) -> Vec<f64> {
    let n = positions.len();
    let dist = |i: usize, j: usize| ((positions[i][0] - positions[j][0]) as f64).hypot((positions[i][1] - positions[j][1]) as f64);
    let threshold = sparsity_threshold.unwrap_or(0.5 / (n.max(1) as f64).sqrt());
    let mut w = vec![0.0f64; z_list_um.len() * n * n];
    let normalise = |w: &mut [f64]| {
        for j in 0..n {
            let norm = (0..n).map(|i| w[i * n + j] * w[i * n + j]).sum::<f64>().sqrt();
            for i in 0..n {
                let v = w[i * n + j] / norm;
                w[i * n + j] = if v.is_finite() { v } else { 0.0 };
            }
        }
    };
    for (zi, &z) in z_list_um.iter().enumerate() {
        let block = &mut w[zi * n * n..(zi + 1) * n * n];
        for i in 0..n {
            for j in 0..n {
                block[i * n + j] = (-(dist(i, j).powi(2) + z * z).sqrt() / sigma_3d).exp();
            }
        }
        normalise(block);
        block.iter_mut().for_each(|v| {
            if *v < threshold {
                *v = 0.0;
            }
        });
        normalise(block);
    }
    w
}

/// A matched-filtering peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchedPeak {
    /// Global sample of the peak (`i + nbefore`).
    pub sample: u64,
    pub channel: u32,
    /// Depth index of the winning source.
    pub z: u32,
    /// The trace's value at the peak.
    pub amplitude: f32,
}

/// The matched filter of one probe and prototype (module docs).
pub struct MatchedFilter {
    opts: MatchedFilterOptions,
    channels: usize,
    templates: usize,
    rows: usize,
    nbefore: usize,
    len: usize,
    prototype: Handle,
    offsets: Handle,
    cols: Handle,
    weights: Handle,
    nnz: usize,
    neighbours: Vec<bool>,
    thresholds: Vec<f64>,
    heights: Option<Handle>,
}

impl MatchedFilter {
    /// `prototype`: the waveform (its extremum at `nbefore`, negative for `peak_sign` neg).
    ///
    /// # Panics
    ///
    /// If the prototype's sample at `nbefore` has the wrong sign for `peak_sign`.
    pub fn new(client: &Client, positions: &[[f32; 2]], prototype: &[f32], nbefore: usize, opts: MatchedFilterOptions) -> Self {
        let n = positions.len();
        let len = prototype.len();
        match opts.peak_sign {
            PeakSign::Neg => assert!(prototype[nbefore] < 0.0, "Prototype should have a negative peak"),
            PeakSign::Pos => assert!(prototype[nbefore] > 0.0, "Prototype should have a positive peak"),
            PeakSign::Both => {}
        }
        // Upstream correlates with the flipped prototype by convolution: a correlation with it
        let norm = prototype.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        // Neg: a spike's correlation with the negative prototype is a positive peak
        let proto: Vec<f32> = prototype.iter().map(|&v| (v as f64 / norm) as f32).collect();
        let w = convolution_weights(positions, &opts.z_list_um, opts.sigma_3d, opts.sparsity_threshold);
        let nz = opts.z_list_um.len();
        let both = opts.peak_sign == PeakSign::Both;
        let templates = if both { 2 * n } else { n };
        let rows = nz * templates;
        // CSR rows (z, t): t's weights over channels j; the doubled templates negated
        let (mut offsets, mut cols, mut weights) = (vec![0u32], Vec::new(), Vec::new());
        for z in 0..nz {
            for t in 0..templates {
                let (i, sign) = if t < n { (t, 1.0) } else { (t - n, -1.0) };
                for j in 0..n {
                    let v = w[z * n * n + i * n + j];
                    if v != 0.0 {
                        cols.push(j as u32);
                        weights.push((sign * v) as f32);
                    }
                }
                offsets.push(cols.len() as u32);
            }
        }
        let neighbours = (0..n * n)
            .map(|e| {
                let (a, b) = (positions[e / n], positions[e % n]);
                (a[0] - b[0]).hypot(a[1] - b[1]) <= opts.radius_um
            })
            .collect();
        let nnz = cols.len();
        Self {
            channels: n,
            templates,
            rows,
            nbefore,
            len,
            prototype: buffer::upload(client, &proto),
            offsets: buffer::upload(client, &offsets),
            cols: buffer::upload(client, if cols.is_empty() { &[0u32][..] } else { &cols }),
            weights: buffer::upload(client, if weights.is_empty() { &[0.0f32][..] } else { &weights }),
            nnz: nnz.max(1),
            neighbours,
            thresholds: Vec::new(),
            heights: None,
            opts,
        }
    }

    /// Rows of the filtered output: depths × templates.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Samples of context each side of a window's own samples.
    pub fn margin(&self) -> usize {
        self.opts.exclude_sweep + self.len + 1
    }

    /// The filtered rows of the `[channels, samples]` device trace: `[rows, samples]`, column `i`
    /// the correlation starting at sample `i` (0 where the prototype runs past the end).
    pub fn filter(&self, client: &Client, trace: &Handle, samples: usize) -> Handle {
        let n = self.channels;
        let corr = buffer::empty::<f32>(client, (n * samples).max(1));
        let total = n * samples;
        let geom = LaunchGeometry::elementwise(client, total);
        // SAFETY: every array is passed with the length it was created with
        unsafe {
            correlate_prototype_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(trace.clone(), total),
                BufferArg::from_raw_parts(self.prototype.clone(), self.len),
                BufferArg::from_raw_parts(corr.clone(), total),
                samples as u32,
                self.len as u32,
                total as u32,
            );
        }
        let total = self.rows * samples;
        let out = buffer::empty::<f32>(client, total.max(1));
        let geom = LaunchGeometry::elementwise(client, total);
        // SAFETY: as above
        unsafe {
            spatial_sum_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(corr, n * samples),
                BufferArg::from_raw_parts(self.offsets.clone(), self.rows + 1),
                BufferArg::from_raw_parts(self.cols.clone(), self.nnz),
                BufferArg::from_raw_parts(self.weights.clone(), self.nnz),
                BufferArg::from_raw_parts(out.clone(), total),
                samples as u32,
                total as u32,
            );
        }
        out
    }

    /// Thresholds from random chunks of the preprocessed recording (device buffers `[channels,
    /// samples]` with their lengths): each row's MAD over the valid correlations of every chunk.
    pub fn fit_thresholds(&mut self, client: &Client, chunks: &[(Handle, usize)]) {
        let mut per_row: Vec<Vec<f32>> = vec![Vec::new(); self.rows];
        for (h, samples) in chunks {
            let valid = (samples + 1).saturating_sub(self.len);
            if valid == 0 {
                continue;
            }
            let out = buffer::download::<f32>(client, self.filter(client, h, *samples));
            for (r, row) in per_row.iter_mut().enumerate() {
                row.extend_from_slice(&out[r * samples..r * samples + valid]);
            }
        }
        self.thresholds = per_row
            .into_iter()
            .map(|mut v| {
                let med = median(&mut v);
                let mut dev: Vec<f32> = v.iter().map(|x| (x - med).abs()).collect();
                median(&mut dev) as f64 / MAD_TO_SIGMA * self.opts.detect_threshold
            })
            .collect();
        let heights: Vec<f32> = self.thresholds.iter().map(|&t| t as f32).collect();
        self.heights = Some(buffer::upload(client, &heights));
    }

    /// The fitted thresholds, per row (`[depths, templates]`).
    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }

    /// Peaks of the device trace whose sample lies in `emit` (local samples; `global_offset`: global
    /// sample of local sample 0), in sample order. Candidates within [`Self::margin`] compete.
    ///
    /// # Panics
    ///
    /// Before [`Self::fit_thresholds`].
    pub fn detect(&self, client: &Client, trace: &Handle, samples: usize, emit: Range<usize>, global_offset: u64) -> Vec<MatchedPeak> {
        let heights = self.heights.as_ref().expect("fit_thresholds first");
        if emit.is_empty() || samples < self.len {
            return Vec::new();
        }
        let filtered = self.filter(client, trace, samples);
        // Correlation index i ↔ peak sample i + nbefore
        let sweep = self.opts.exclude_sweep + 1;
        let lo = emit.start.saturating_sub(self.nbefore + sweep);
        let hi = (emit.end.saturating_sub(self.nbefore) + sweep).min(samples + 1 - self.len);
        if lo >= hi {
            return Vec::new();
        }
        let found = find_peak_candidates::<f32>(client, &filtered, heights, self.rows, samples, lo..hi, Polarity::Positive);
        let mut cands: Vec<Candidate> = Vec::new();
        for r in 0..self.rows {
            let (z, t) = (r / self.templates, r % self.templates);
            let (idx, vals) = found.channel(r);
            for (&i, &v) in idx.iter().zip(vals) {
                cands.push(Candidate { sample: i as u64, row: t as u32, score: v as f64 / self.thresholds[r], tiebreak: z as u32 });
            }
        }
        cands.sort_by_key(|c| (c.sample, c.tiebreak, c.row));
        let n = self.channels;
        let keep = locally_exclusive(&cands, self.opts.exclude_sweep as u64, |a, b| self.neighbours[(a as usize % n) * n + b as usize % n], true);
        let kept: Vec<Candidate> = cands
            .into_iter()
            .zip(keep)
            .filter(|(c, k)| *k && emit.contains(&(c.sample as usize + self.nbefore)))
            .map(|(c, _)| c)
            .collect();
        if kept.is_empty() {
            return Vec::new();
        }
        let chans: Vec<u32> = kept.iter().map(|c| c.row % n as u32).collect();
        let at: Vec<u32> = kept.iter().map(|c| (c.sample as usize + self.nbefore) as u32).collect();
        let amps = gather_points(client, trace, &chans, &at, n * samples, samples);
        kept.iter()
            .zip(amps)
            .map(|(c, a)| MatchedPeak { sample: global_offset + c.sample + self.nbefore as u64, channel: c.row % n as u32, z: c.tiebreak, amplitude: a })
            .collect()
    }
}

/// Values of a `[_, samples]` device buffer at `(channels[i], at[i])`.
pub fn gather_points(client: &Client, x: &Handle, channels: &[u32], at: &[u32], len: usize, samples: usize) -> Vec<f32> {
    let k = channels.len();
    if k == 0 {
        return Vec::new();
    }
    let out = buffer::empty::<f32>(client, k);
    let geom = LaunchGeometry::elementwise(client, k);
    // SAFETY: `x` holds `len` values; the points lie inside it
    unsafe {
        gather_points_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(x.clone(), len),
            BufferArg::from_raw_parts(buffer::upload(client, channels), k),
            BufferArg::from_raw_parts(buffer::upload(client, at), k),
            BufferArg::from_raw_parts(out.clone(), k),
            samples as u32,
            k as u32,
        );
    }
    buffer::download::<f32>(client, out)
}

/// NumPy's median (`v` reordered).
fn median(v: &mut [f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let (lower, mid, _) = v.select_nth_unstable_by(n / 2, f32::total_cmp);
    let mid = *mid;
    if n % 2 == 1 { mid } else { 0.5 * (lower.iter().copied().fold(f32::NEG_INFINITY, f32::max) + mid) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;

    #[test]
    fn weights_follow_upstream() {
        let pos: Vec<[f32; 2]> = (0..6).map(|c| [0.0, 20.0 * c as f32]).collect();
        let w = convolution_weights(&pos, &[0.0, 120.0], 2.5, None);
        let n = 6;
        // At depth 0 a source sits on its channel: exp(−20/2.5) ≈ 3e-4 is below 0.5/√6
        for j in 0..n {
            for i in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((w[i * n + j] - want).abs() < 1e-12, "z 0 [{i},{j}] = {}", w[i * n + j]);
            }
        }
        // At 120 µm the neighbours count (relative weight e^(−(√(20² + 120²) − 120) / 2.5) ≈ 0.52 at
        // 20 µm); every column has unit norm
        for j in 0..n {
            let col: Vec<f64> = (0..n).map(|i| w[n * n + i * n + j]).collect();
            assert!((col.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(col[j] > 0.5 && (j == 0 || col[j - 1] > 0.2), "{col:?}");
        }
    }

    /// A planted prototype on one channel is found at its peak sample, once.
    #[test]
    fn finds_planted_waveforms() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (m, n) = (5usize, 6000usize);
        let pos: Vec<[f32; 2]> = (0..m).map(|c| [0.0, 25.0 * c as f32]).collect();
        let nbefore = 8;
        let proto: Vec<f32> = (0..30).map(|k| -(-((k as f32 - nbefore as f32) / 3.0).powi(2)).exp() + 0.3 * (-((k as f32 - 16.0) / 4.0).powi(2)).exp()).collect();
        let mut state = 11u64;
        let mut x: Vec<f32> = (0..m * n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 0.6
            })
            .collect();
        let planted: Vec<(usize, usize)> = (0..12).map(|k| (200 + k * 450, k % m)).collect();
        for &(t, c) in &planted {
            for (k, p) in proto.iter().enumerate() {
                x[c * n + t - nbefore + k] += 8.0 * p;
            }
        }
        let trace = buffer::upload(&client, &x);
        let opts = MatchedFilterOptions { exclude_sweep: 20, ..Default::default() };
        let mut mf = MatchedFilter::new(&client, &pos, &proto, nbefore, opts);
        mf.fit_thresholds(&client, &[(trace.clone(), n)]);
        let peaks = mf.detect(&client, &trace, n, 50..n - 50, 0);
        let got: Vec<(u64, u32)> = peaks.iter().map(|p| (p.sample, p.channel)).collect();
        let want: Vec<(u64, u32)> = planted.iter().map(|&(t, c)| (t as u64, c as u32)).collect();
        assert_eq!(got, want, "{peaks:?}");
        assert!(peaks.iter().all(|p| p.amplitude < -6.0));
    }
}
