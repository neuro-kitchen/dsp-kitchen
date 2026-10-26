//! SpikeInterface's `circus-omp` template matching (`sortingcomponents/matching/circus.py`
//! `CircusOMPPeeler`, MIT; numpy engine), the matching of SpyKING CIRCUS 2.
//!
//! **Preparation** ([`CircusOmp::new`]): every template (`[width, channels]`, zeros off its
//! sparsity) is compressed to rank `rank` by SVD (`U S Vᵀ`), normalised by the norm of the
//! compressed template on its own channels. Units overlap when their channels intersect; for each
//! overlapping pair, the overlap of their normalised templates at every lag (`2·width − 1`, upstream's
//! formula: unit `i`'s compressed template, in time order, on unit `j`'s channels, projected on `j`'s
//! spatial components and scaled, convolved per rank with `j`'s time-reversed temporal components).
//!
//! **Scalar products** ([`CircusOmp::scalar_products`], device): for every template and start sample
//! `i`, `Σ_r s_r Σ_k U[k, r] / norm · (Vᵀ_r · x)[i + k]`: the data's correlation with the normalised
//! compressed template, as a spatial matrix product and a short temporal correlation.
//!
//! **Greedy loop** ([`CircusOmp::match_scalar_products`], host, one chunk): each round takes, at
//! every sample, the best template; the samples whose best product is a local maximum within
//! `±vicinity` (`2·width`; `scipy.ndimage.maximum_filter`, zeros outside) and whose product / norm
//! exceeds 0.25 are peaks. Each peak in turn extends the Cholesky factor `M` of the selected atoms
//! (its row: the overlaps with the selected atoms closer than `width`, solved against the atoms closer
//! than `vicinity`; a dependent atom ends the round), and the amplitudes of the atoms closer than
//! `vicinity` are re-solved from the original products; every changed amplitude subtracts its change
//! times the overlaps from the products. The loop stops after `max_failures` rounds without a new
//! valid amplitude (`amplitudes.0 < a < amplitudes.1`). Spikes: every valid (template, sample), at
//! `sample + nbefore`.
//!
//! Upstream runs the loop on chunks with a margin of `2·width` each side and keeps the spikes inside;
//! the chunks are independent, so they can run in parallel.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::linalg::{matmul, symmetric_eigen_cpu, MatrixView};
use dsp_core::compute::LaunchGeometry;

use super::kernels::omp_scalar_products_kernel;

/// Settings of [`CircusOmp`] (upstream defaults in [`Default`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OmpOptions {
    /// Valid amplitudes lie strictly between these.
    pub amplitudes: (f32, f32),
    pub max_failures: usize,
    pub rank: usize,
    /// In template widths.
    pub vicinity: usize,
}

impl Default for OmpOptions {
    fn default() -> Self {
        Self { amplitudes: (0.6, f32::INFINITY), max_failures: 5, rank: 5, vicinity: 2 }
    }
}

/// A matched spike: template, sample (chunk-local `i + nbefore`), amplitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OmpSpike {
    pub template: u32,
    pub sample: usize,
    pub amplitude: f32,
}

/// The prepared templates (module docs).
pub struct CircusOmp {
    opts: OmpOptions,
    k: usize,
    width: usize,
    channels: usize,
    rank: usize,
    nbefore: usize,
    norms: Vec<f32>,
    /// Per template: the overlapping templates (ascending) and their overlaps `[n, 2·width − 1]`.
    overlap_units: Vec<Vec<usize>>,
    overlaps: Vec<Vec<f32>>,
    /// Per template: position of each template in `overlap_units` (`usize::MAX`: none).
    overlap_table: Vec<Vec<usize>>,
    spatial: Handle,
    temporal: Handle,
}

impl CircusOmp {
    /// `templates`: `[k, width, channels]` dense (zeros off `sparsity`, `[k, channels]`).
    pub fn new(client: &Client, templates: &[f32], sparsity: &[bool], k: usize, width: usize, channels: usize, nbefore: usize, opts: OmpOptions) -> Self {
        let (w, m) = (width, channels);
        let rank = opts.rank.min(w).min(m);
        // Rank-r SVD of each template: U (w × r), s (r), Vt (r × m)
        let mut u_all = vec![0.0f64; k * w * rank];
        let mut s_all = vec![0.0f64; k * rank];
        let mut vt_all = vec![0.0f64; k * rank * m];
        for t in 0..k {
            let a = &templates[t * w * m..(t + 1) * w * m];
            let mut g = vec![0.0f64; w * w];
            for p in 0..w {
                for q in p..w {
                    let v: f64 = (0..m).map(|c| a[p * m + c] as f64 * a[q * m + c] as f64).sum();
                    g[p * w + q] = v;
                    g[q * w + p] = v;
                }
            }
            let eig = symmetric_eigen_cpu(&g, w);
            for r in 0..rank {
                let sigma = eig.values[r].max(0.0).sqrt();
                s_all[t * rank + r] = sigma;
                for p in 0..w {
                    u_all[(t * w + p) * rank + r] = eig.vectors[p * w + r];
                }
                if sigma > 0.0 {
                    for c in 0..m {
                        let v: f64 = (0..w).map(|p| a[p * m + c] as f64 * eig.vectors[p * w + r]).sum();
                        vt_all[(t * rank + r) * m + c] = v / sigma;
                    }
                }
            }
        }
        // Compressed templates, their norms on their own channels
        let recon = |t: usize, p: usize, c: usize| -> f64 { (0..rank).map(|r| u_all[(t * w + p) * rank + r] * s_all[t * rank + r] * vt_all[(t * rank + r) * m + c]).sum() };
        let norms: Vec<f64> = (0..k)
            .map(|t| {
                let mut ss = 0.0;
                for p in 0..w {
                    for c in 0..m {
                        if sparsity[t * m + c] {
                            ss += recon(t, p, c).powi(2);
                        }
                    }
                }
                ss.sqrt()
            })
            .collect();
        // Overlaps (module docs)
        let lags = 2 * w - 1;
        let mut overlap_units = Vec::with_capacity(k);
        let mut overlaps = Vec::with_capacity(k);
        let mut overlap_table = Vec::with_capacity(k);
        let full: Vec<Vec<f64>> = (0..k).map(|t| (0..w * m).map(|e| recon(t, e / m, e % m) / norms[t]).collect()).collect();
        for i in 0..k {
            let units: Vec<usize> = (0..k).filter(|&j| (0..m).any(|c| sparsity[i * m + c] && sparsity[j * m + c])).collect();
            let mut rows = vec![0.0f32; units.len() * lags];
            for (row, &j) in units.iter().enumerate() {
                let chans: Vec<usize> = (0..m).filter(|&c| sparsity[j * m + c]).collect();
                for r in 0..rank {
                    // visible[kk] = s_j,r · Σ_c template_i[kk, c] Vt_j[r, c] (upstream reverses the
                    // already reversed temporal components back: template_i in time order)
                    let visible: Vec<f64> = (0..w)
                        .map(|kk| s_all[j * rank + r] * chans.iter().map(|&c| full[i][kk * m + c] * vt_all[(j * rank + r) * m + c]).sum::<f64>())
                        .collect();
                    // temporal_j flipped and normalised
                    let tau: Vec<f64> = (0..w).map(|kk| u_all[(j * w + (w - 1 - kk)) * rank + r] / norms[j]).collect();
                    for d in 0..lags {
                        let mut acc = 0.0;
                        for kk in 0..w {
                            if d >= kk && d - kk < w {
                                acc += visible[kk] * tau[d - kk];
                            }
                        }
                        rows[row * lags + d] += acc as f32;
                    }
                }
            }
            let mut table = vec![usize::MAX; k];
            for (pos, &j) in units.iter().enumerate() {
                table[j] = pos;
            }
            overlap_units.push(units);
            overlaps.push(rows);
            overlap_table.push(table);
        }
        // Device: Vt stacked [(t, r), m], and the temporal weights s · U / norm, [t, w, r]
        let spatial: Vec<f32> = vt_all.iter().map(|&v| v as f32).collect();
        let temporal: Vec<f32> = (0..k * w * rank)
            .map(|e| {
                let (t, p, r) = (e / (w * rank), (e / rank) % w, e % rank);
                (s_all[t * rank + r] * u_all[(t * w + p) * rank + r] / norms[t]) as f32
            })
            .collect();
        let up = |v: &[f32]| buffer::upload(client, if v.is_empty() { &[0.0f32][..] } else { v });
        Self {
            opts,
            k,
            width: w,
            channels: m,
            rank,
            nbefore,
            norms: norms.iter().map(|&n| n as f32).collect(),
            overlap_units,
            overlaps,
            overlap_table,
            spatial: up(&spatial),
            temporal: up(&temporal),
        }
    }

    /// Samples of context each side of a chunk (upstream `get_margin`): `2 · width`.
    pub fn margin(&self) -> usize {
        2 * self.width
    }

    /// Scalar products of the `[channels, samples]` device chunk: `[k, samples − width + 1]` on the
    /// host (module docs).
    pub fn scalar_products(&self, client: &Client, x: &Handle, samples: usize) -> Vec<f32> {
        let (k, w, m, r) = (self.k, self.width, self.channels, self.rank);
        if samples < w || k == 0 {
            return Vec::new();
        }
        let peaks = samples - w + 1;
        let y = buffer::empty::<f32>(client, k * r * samples);
        matmul::<f32>(
            client,
            &MatrixView::row_major(&self.spatial, k * r * m, k * r, m),
            &MatrixView::row_major(x, m * samples, m, samples),
            &y,
            k * r * samples,
        );
        let total = k * peaks;
        let out = buffer::empty::<f32>(client, total);
        let geom = LaunchGeometry::elementwise(client, total);
        // SAFETY: every array is passed with the length it was created with
        unsafe {
            omp_scalar_products_kernel::launch::<f32>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(y, k * r * samples),
                BufferArg::from_raw_parts(self.temporal.clone(), k * w * r),
                BufferArg::from_raw_parts(out.clone(), total),
                samples as u32,
                w as u32,
                r as u32,
                peaks as u32,
                total as u32,
            );
        }
        buffer::download::<f32>(client, out)
    }

    /// The greedy loop on one chunk's scalar products (`[k, peaks]`; module docs).
    pub fn match_scalar_products(&self, mut sp: Vec<f32>) -> Vec<OmpSpike> {
        let (k, w) = (self.k, self.width);
        if k == 0 || sp.is_empty() {
            return Vec::new();
        }
        let p = sp.len() / k;
        let full = sp.clone();
        let vicinity = self.opts.vicinity * w;
        let neighbor = w - 1;
        let eps = f32::EPSILON;
        let (amin, amax) = self.opts.amplitudes;
        let mut final_amp = vec![0.0f32; k * p];
        let mut touched: Vec<usize> = Vec::new();
        let mut cap = k.max(1);
        let mut mm = vec![0.0f32; cap * cap];
        let mut sel_t: Vec<usize> = Vec::new();
        let mut sel_i: Vec<usize> = Vec::new();
        let mut in_vicinity: Vec<usize> = Vec::new();
        let mut num_valids = 0usize;
        let mut failures = self.opts.max_failures;
        let mut best = vec![0usize; p];
        let mut products = vec![0.0f32; p];
        let mut filtered = vec![0.0f32; p];
        loop {
            for i in 0..p {
                let (mut b, mut v) = (0usize, sp[i]);
                for t in 1..k {
                    if sp[t * p + i] > v {
                        v = sp[t * p + i];
                        b = t;
                    }
                }
                best[i] = b;
                products[i] = v;
            }
            maximum_filter(&products, vicinity, &mut filtered);
            let peaks: Vec<usize> = (0..p)
                .filter(|&i| products[i] / self.norms[best[i]] > 0.25 && (products[i] - filtered[i]).abs() < 1e-9)
                .collect();
            if peaks.is_empty() {
                break;
            }
            for &peak in &peaks {
                let b = best[peak];
                let ns = sel_t.len();
                if ns > 0 {
                    if ns == cap {
                        let mut grown = vec![0.0f32; 4 * cap * cap];
                        for row in 0..cap {
                            grown[row * 2 * cap..row * 2 * cap + cap].copy_from_slice(&mm[row * cap..(row + 1) * cap]);
                        }
                        cap *= 2;
                        mm = grown;
                    }
                    // Overlaps with the atoms closer than a width
                    for s in 0..ns {
                        let dt = sel_i[s] as i64 - peak as i64;
                        if dt.unsigned_abs() < w as u64 {
                            let pos = self.overlap_table[b][sel_t[s]];
                            if pos != usize::MAX {
                                let line = (neighbor as i64 + dt) as usize;
                                mm[ns * cap + s] = self.overlaps[b][pos * (2 * w - 1) + line];
                            }
                        }
                    }
                    in_vicinity = (0..ns).filter(|&s| ((sel_i[s] as i64 - peak as i64).unsigned_abs() as usize) < vicinity).collect();
                    if !in_vicinity.is_empty() {
                        let mut rhs: Vec<f32> = in_vicinity.iter().map(|&s| mm[ns * cap + s]).collect();
                        forward_substitute(&mm, cap, &in_vicinity, &mut rhs);
                        for (q, &s) in in_vicinity.iter().enumerate() {
                            mm[ns * cap + s] = rhs[q];
                        }
                        let v: f32 = rhs.iter().map(|x| x * x).sum();
                        let lkk = 1.0 - v;
                        if lkk <= eps {
                            break;
                        }
                        mm[ns * cap + ns] = lkk.sqrt();
                    } else {
                        mm[ns * cap + ns] = 1.0;
                    }
                } else {
                    mm[0] = 1.0;
                }
                sel_t.push(b);
                sel_i.push(peak);
                let ns = sel_t.len();
                in_vicinity.push(ns - 1);
                let rhs0: Vec<f32> = in_vicinity.iter().map(|&s| full[sel_t[s] * p + sel_i[s]]).collect();
                let mut amps = rhs0.clone();
                forward_substitute(&mm, cap, &in_vicinity, &mut amps);
                backward_substitute(&mm, cap, &in_vicinity, &mut amps);
                for (q, &s) in in_vicinity.iter().enumerate() {
                    let (t, i) = (sel_t[s], sel_i[s]);
                    let new = amps[q] / self.norms[t];
                    let diff = new - final_amp[t * p + i];
                    final_amp[t * p + i] = new;
                    touched.push(t * p + i);
                    if diff.abs() > eps {
                        self.subtract(&mut sp, p, t, i, diff * self.norms[t]);
                    }
                }
            }
            touched.sort_unstable();
            touched.dedup();
            let valids = touched.iter().filter(|&&e| final_amp[e] > amin && final_amp[e] < amax).count();
            if valids > num_valids {
                failures = self.opts.max_failures;
            } else {
                failures = failures.saturating_sub(1);
            }
            num_valids = valids;
            if failures == 0 {
                break;
            }
        }
        let mut spikes: Vec<OmpSpike> = touched
            .iter()
            .filter(|&&e| final_amp[e] > amin && final_amp[e] < amax)
            .map(|&e| OmpSpike { template: (e / p) as u32, sample: e % p + self.nbefore, amplitude: final_amp[e] })
            .collect();
        spikes.sort_by_key(|s| (s.sample, s.template));
        spikes
    }

    /// `sp[j, ·] −= amp · overlap(t, j)` around `peak` for every template `j` overlapping `t`.
    fn subtract(&self, sp: &mut [f32], p: usize, t: usize, peak: usize, amp: f32) {
        let w = self.width;
        let lags = 2 * w - 1;
        let tmp = peak as i64 - (w as i64 - 1);
        let (lo, hi) = (tmp.max(0) as usize, (peak + w).min(p));
        for (row, &j) in self.overlap_units[t].iter().enumerate() {
            let ov = &self.overlaps[t][row * lags..(row + 1) * lags];
            for idx in lo..hi {
                sp[j * p + idx] -= amp * ov[(idx as i64 - tmp) as usize];
            }
        }
    }
}

/// `scipy.ndimage.maximum_filter(x, size, mode="constant", cval=0)` in one dimension: the window
/// covers offsets `−size/2 ..= size − 1 − size/2`.
fn maximum_filter(x: &[f32], size: usize, out: &mut [f32]) {
    let n = x.len();
    let (before, after) = (size / 2, size.saturating_sub(1) - size / 2);
    for i in 0..n {
        let mut m = if i < before || i + after >= n { 0.0f32 } else { f32::NEG_INFINITY };
        let lo = i.saturating_sub(before);
        let hi = (i + after).min(n - 1);
        for &v in &x[lo..=hi] {
            m = m.max(v);
        }
        out[i] = m;
    }
}

/// Solves `L y = b` for the lower-triangular `L = M[idx, idx]` (`M` row-major, `cap` wide).
fn forward_substitute(mm: &[f32], cap: usize, idx: &[usize], b: &mut [f32]) {
    for a in 0..idx.len() {
        let mut v = b[a];
        for c in 0..a {
            v -= mm[idx[a] * cap + idx[c]] * b[c];
        }
        b[a] = v / mm[idx[a] * cap + idx[a]];
    }
}

/// Solves `Lᵀ x = y` for the same `L` (the second half of LAPACK `potrs`).
fn backward_substitute(mm: &[f32], cap: usize, idx: &[usize], b: &mut [f32]) {
    for a in (0..idx.len()).rev() {
        let mut v = b[a];
        for c in a + 1..idx.len() {
            v -= mm[idx[c] * cap + idx[a]] * b[c];
        }
        b[a] = v / mm[idx[a] * cap + idx[a]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;

    #[test]
    fn maximum_filter_matches_scipy_windows() {
        let x = [1.0f32, 3.0, 2.0, 5.0, 4.0];
        let mut out = [0.0f32; 5];
        // size 2: offsets −1..=0
        maximum_filter(&x, 2, &mut out);
        assert_eq!(out, [1.0, 3.0, 3.0, 5.0, 5.0]);
        // size 4: offsets −2..=1, zeros outside
        maximum_filter(&[-1.0, -2.0, -3.0], 4, &mut out[..3]);
        assert_eq!(&out[..3], &[0.0, 0.0, 0.0]);
    }

    /// Two templates planted with known amplitudes, overlapping in time, are recovered.
    #[test]
    fn recovers_planted_templates() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (w, m, n) = (20usize, 3usize, 2000usize);
        let nbefore = 6;
        let shape = |s: usize, a: f32, b: f32| -> f32 { let t = s as f32 - nbefore as f32; -a * (-(t / 2.0).powi(2)).exp() + b * (-((t - 5.0) / 3.0).powi(2)).exp() };
        let mut tmpl = vec![0.0f32; 2 * w * m];
        for s in 0..w {
            tmpl[s * m] = shape(s, 10.0, 3.0);
            tmpl[s * m + 1] = shape(s, 6.0, 2.0);
            tmpl[w * m + s * m + 1] = shape(s, 4.0, 4.0);
            tmpl[w * m + s * m + 2] = shape(s, 9.0, 1.0);
        }
        let sparsity = vec![true, true, false, false, true, true];
        let omp = CircusOmp::new(&client, &tmpl, &sparsity, 2, w, m, nbefore, OmpOptions::default());
        // Spikes: template, start sample, amplitude
        let planted = [(0usize, 200usize, 1.0f32), (1, 210, 1.3), (0, 700, 0.9), (1, 1200, 1.1), (0, 1205, 1.0)];
        let mut x = vec![0.0f32; m * n];
        for &(t, start, a) in &planted {
            for s in 0..w {
                for c in 0..m {
                    x[c * n + start + s] += a * tmpl[(t * w + s) * m + c];
                }
            }
        }
        let sp = omp.scalar_products(&client, &buffer::upload(&client, &x), n);
        let spikes = omp.match_scalar_products(sp);
        let got: Vec<(u32, usize)> = spikes.iter().map(|s| (s.template, s.sample)).collect();
        let want: Vec<(u32, usize)> = planted.iter().map(|&(t, start, _)| (t as u32, start + nbefore)).collect();
        let mut want_sorted = want.clone();
        want_sorted.sort_by_key(|&(t, s)| (s, t));
        assert_eq!(got, want_sorted, "{spikes:?}");
        for s in &spikes {
            let a = planted.iter().find(|p| p.0 as u32 == s.template && p.1 + nbefore == s.sample).unwrap().2;
            assert!((s.amplitude - a).abs() < 1e-2, "{s:?} vs {a}");
        }
    }
}
