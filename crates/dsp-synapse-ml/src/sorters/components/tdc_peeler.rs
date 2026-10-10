//! SpikeInterface's `tdc-peeler` template matching (`sortingcomponents/matching/tdc_peeler.py`
//! `TridesclousPeeler`, MIT; static, no motion), the matching of Tridesclous 2. On the host, one
//! chunk at a time (chunks are independent: run them in parallel).
//!
//! A chunk (with its margins) is peeled in **levels**: up to `max_peeler_loop` levels with the fast
//! detector (`locally_exclusive`), then one with the fine one (matched filtering), stopping early when
//! a level finds nothing. A level:
//! 1. detects peaks on the residual, ordered by decreasing `|amplitude|`;
//! 2. for each peak: the candidate units are those whose main channel lies within `cluster_radius_um`
//!    of the peak channel; the best is the one whose short template (`ms_before`, `ms_after`) is
//!    closest (squared distance over the union of the candidates' channels, a candidate's template 0
//!    off its own channels); the best shift within `±sample_shift`; a peak whose (sample, unit) a
//!    previous level already found is dropped; the units of the later peaks nearby (within
//!    `max(nbefore, nafter)` and `amplitude_fitting_radius_um`) are guessed the same way, and the
//!    amplitude fitted by least squares with them (on the unit's channels); amplitudes within
//!    `amplitude_limits` are kept and the template subtracted, smaller ones dropped, larger ones kept
//!    with amplitude 1 and **not subtracted** (upstream).
//!
//! Upstream quirk kept: the fine detector's prototype is built in a loop over units that reads the
//! last unit's template only (a variable left over from the previous loop): the last unit's waveform
//! on its most negative channel, scaled to −1 at the peak. Its spatial weights use one source depth
//! (50 µm).

use dsp_base::linalg::symmetric_eigen_cpu;

use super::exclusive::{locally_exclusive, Candidate};
use super::matched::convolution_weights;
use super::templates::Templates;

/// Settings of [`TdcPeeler`] (upstream defaults in [`Default`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdcPeelerOptions {
    pub exclude_sweep_ms: f64,
    pub detect_threshold: f64,
    pub detection_radius_um: f32,
    pub cluster_radius_um: f32,
    pub amplitude_fitting_radius_um: f32,
    pub sample_shift: usize,
    pub ms_before: f64,
    pub ms_after: f64,
    pub max_peeler_loop: usize,
    pub amplitude_limits: (f64, f64),
    pub use_fine_detector: bool,
}

impl Default for TdcPeelerOptions {
    fn default() -> Self {
        Self {
            exclude_sweep_ms: 0.8,
            detect_threshold: 5.0,
            detection_radius_um: 80.0,
            cluster_radius_um: 150.0,
            amplitude_fitting_radius_um: 150.0,
            sample_shift: 2,
            ms_before: 0.5,
            ms_after: 0.8,
            max_peeler_loop: 2,
            amplitude_limits: (0.7, 1.4),
            use_fine_detector: true,
        }
    }
}

/// A spike found by the peeler: chunk-local sample, unit, amplitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdcSpike {
    pub sample: i64,
    pub channel: usize,
    pub unit: i64,
    pub amplitude: f64,
}

/// The fine detector's matched filter on the host.
#[derive(Debug, Clone)]
pub struct FineFilter {
    /// Normalised prototype (correlation weights).
    pub prototype: Vec<f32>,
    pub nbefore: usize,
    /// Per template row `t` (one depth): `(channel, weight)` pairs.
    pub rows: Vec<Vec<(usize, f32)>>,
    /// Per row, the absolute threshold.
    pub thresholds: Vec<f64>,
}

/// The prepared peeler (module docs).
#[derive(Debug, Clone)]
pub struct TdcPeeler {
    opts: TdcPeelerOptions,
    templates: Templates,
    nbefore: usize,
    nbefore_short: usize,
    /// Short window: template samples `s0..s0 + width_short`.
    s0: usize,
    width_short: usize,
    norms: Vec<f64>,
    possible_clusters: Vec<Vec<usize>>,
    near: Vec<bool>,
    detect_near: Vec<bool>,
    thresholds: Vec<f64>,
    sweep: usize,
    fine: Option<FineFilter>,
}

/// The fine detector's prototype (module docs: upstream's quirk) from the last unit's template.
pub fn fine_prototype(t: &Templates, nbefore: usize) -> Option<Vec<f32>> {
    let (w, m) = (t.width, t.channels);
    let u = t.len().checked_sub(1)?;
    let chans: Vec<usize> = (0..m).filter(|&c| t.sparsity[u * m + c]).collect();
    // argmin over the sparse columns at nbefore (the first on ties)
    let ch = *chans.iter().min_by(|&&a, &&b| t.data[(u * w + nbefore) * m + a].total_cmp(&t.data[(u * w + nbefore) * m + b]))?;
    let peak = t.data[(u * w + nbefore) * m + ch];
    if peak == 0.0 {
        return None;
    }
    let p: Vec<f32> = (0..w).map(|s| t.data[(u * w + s) * m + ch] / peak.abs()).collect();
    let scale = p[nbefore].abs();
    Some(p.iter().map(|v| v / scale).collect())
}

impl FineFilter {
    /// The prototype normalised, `exponential_3d` weights at depth 50 µm; thresholds from
    /// `noise_mad` (the rows' MADs on random chunks, fitted elsewhere) × `detect_threshold`.
    pub fn new(prototype: &[f32], nbefore: usize, positions: &[[f32; 2]], row_mad: &[f64], detect_threshold: f64) -> Self {
        let n = positions.len();
        let norm = prototype.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let w = convolution_weights(positions, &[50.0], 2.5, None);
        let rows = (0..n)
            .map(|t| (0..n).filter_map(|j| { let v = w[t * n + j]; (v != 0.0).then_some((j, v as f32)) }).collect())
            .collect();
        Self {
            prototype: prototype.iter().map(|&v| (v as f64 / norm) as f32).collect(),
            nbefore,
            rows,
            thresholds: row_mad.iter().map(|m| m * detect_threshold).collect(),
        }
    }
}

impl FineFilter {
    /// [`FineFilter::new`] with each row's noise measured on `chunks` (host `[channels, len]`
    /// buffers): the median absolute deviation (around the median, `/ 0.6745`) of the filtered rows
    /// over the chunks' valid correlations.
    pub fn fit(prototype: &[f32], nbefore: usize, positions: &[[f32; 2]], chunks: &[(Vec<f32>, usize)], detect_threshold: f64) -> Self {
        let unfitted = Self::new(prototype, nbefore, positions, &vec![1.0; positions.len()], 1.0);
        let (m, l) = (positions.len(), unfitted.prototype.len());
        let mut rows: Vec<Vec<f32>> = vec![Vec::new(); m];
        for (x, len) in chunks {
            let valid = (len + 1).saturating_sub(l);
            let corr: Vec<f32> = (0..m * valid).map(|e| { let (c, i) = (e / valid, e % valid); (0..l).map(|k| unfitted.prototype[k] * x[c * len + i + k]).sum() }).collect();
            for (t, row) in rows.iter_mut().enumerate() {
                row.extend((0..valid).map(|i| unfitted.rows[t].iter().map(|&(c, wt)| wt * corr[c * valid + i]).sum::<f32>()));
            }
        }
        let mad: Vec<f64> = rows
            .into_iter()
            .map(|mut r| {
                let med = median_f32(&mut r);
                let mut dev: Vec<f32> = r.iter().map(|v| (v - med as f32).abs()).collect();
                median_f32(&mut dev) / 0.674_489_750_196_081_7
            })
            .collect();
        Self { thresholds: mad.iter().map(|v| v * detect_threshold).collect(), ..unfitted }
    }
}

/// NumPy's median (`v` reordered).
fn median_f32(v: &mut [f32]) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let (lower, mid, _) = v.select_nth_unstable_by(n / 2, f32::total_cmp);
    let mid = *mid as f64;
    if n % 2 == 1 { mid } else { 0.5 * (lower.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64 + mid) }
}

impl TdcPeeler {
    /// `templates`: dense `[k, width, channels]` with sparsity, `nbefore` samples before the peak;
    /// `noise`: per channel; `fine`: the fine detector (with `use_fine_detector`).
    pub fn new(templates: Templates, nbefore: usize, positions: &[[f32; 2]], noise: &[f64], fs: f64, fine: Option<FineFilter>, opts: TdcPeelerOptions) -> Self {
        let (k, w, m) = (templates.len(), templates.width, templates.channels);
        let nafter = w - nbefore;
        let ms = |v: f64| (v * fs / 1000.0) as usize;
        let (nbefore_short, nafter_short) = (ms(opts.ms_before).min(nbefore), ms(opts.ms_after).min(nafter));
        let s0 = nbefore - nbefore_short;
        let dist = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
        // Main channel: the deepest trough
        let main: Vec<usize> = (0..k)
            .map(|u| (0..w * m).fold(0usize, |b, e| if templates.data[u * w * m + e] < templates.data[u * w * m + b] { e } else { b }) % m)
            .collect();
        let possible_clusters = (0..m).map(|c| (0..k).filter(|&u| dist(positions[c], positions[main[u]]) <= opts.cluster_radius_um).collect()).collect();
        let near = (0..m * m).map(|e| dist(positions[e / m], positions[e % m]) <= opts.amplitude_fitting_radius_um).collect();
        let detect_near = (0..m * m).map(|e| dist(positions[e / m], positions[e % m]) <= opts.detection_radius_um).collect();
        let norms = (0..k)
            .map(|u| {
                let mut ss = 0.0f64;
                for s in 0..w {
                    for c in 0..m {
                        if templates.sparsity[u * m + c] {
                            ss += (templates.data[(u * w + s) * m + c] as f64).powi(2);
                        }
                    }
                }
                ss
            })
            .collect();
        Self {
            thresholds: noise.iter().map(|n| n * opts.detect_threshold).collect(),
            sweep: ms(opts.exclude_sweep_ms),
            fine: if opts.use_fine_detector { fine } else { None },
            opts,
            templates,
            nbefore,
            nbefore_short,
            s0,
            width_short: nbefore_short + nafter_short,
            norms,
            possible_clusters,
            near,
            detect_near,
        }
    }

    fn nafter(&self) -> usize {
        self.templates.width - self.nbefore
    }

    /// Samples of context each side of a chunk.
    pub fn margin(&self) -> usize {
        let peeler = 2 * self.nbefore.max(self.nafter());
        let fast = self.sweep + 1;
        let fine = self.fine.as_ref().map_or(0, |f| self.sweep + f.prototype.len() + 1);
        peeler.max(fast).max(fine)
    }

    /// Peels a chunk (`x`: `[channels, samples]` row-major, modified into the residual); spikes in
    /// sample order (chunk-local samples).
    pub fn peel(&self, x: &mut [f32], samples: usize) -> Vec<TdcSpike> {
        let mut all = Vec::new();
        let mut prev: Vec<TdcSpike> = Vec::new();
        let mut level = 0;
        let mut fine = false;
        loop {
            let spikes = self.level(x, samples, &prev, fine);
            level += 1;
            let empty = spikes.is_empty();
            all.extend_from_slice(&spikes);
            prev.extend(spikes);
            if empty || level == self.opts.max_peeler_loop {
                if self.fine.is_some() && !fine {
                    fine = true;
                    level = self.opts.max_peeler_loop - 1;
                    continue;
                }
                break;
            }
        }
        all.sort_by_key(|s| s.sample);
        all
    }

    /// One level (module docs).
    fn level(&self, x: &mut [f32], n: usize, prev: &[TdcSpike], fine: bool) -> Vec<TdcSpike> {
        let (w, m) = (self.templates.width, self.templates.channels);
        let at = |x: &[f32], s: i64, c: usize| x[c * n + s as usize];
        // Detection on the trace minus the detector's margin difference
        let peeler_margin = 2 * self.nbefore.max(self.nafter());
        let detector_margin = if fine { self.sweep + self.fine.as_ref().map_or(0, |f| f.prototype.len()) + 1 } else { self.sweep + 1 };
        let shift = peeler_margin.saturating_sub(detector_margin);
        let (lo, hi) = (shift, n - shift);
        let peaks: Vec<(i64, usize)> = if fine { self.detect_fine(x, n, lo, hi) } else { self.detect_fast(x, n, lo, hi) };
        // Decreasing |amplitude| (upstream reverses an ascending sort)
        let mut order: Vec<usize> = (0..peaks.len()).collect();
        order.sort_by(|&a, &b| at(x, peaks[a].0, peaks[a].1).abs().total_cmp(&at(x, peaks[b].0, peaks[b].1).abs()));
        order.reverse();
        let peaks: Vec<(i64, usize)> = order.iter().map(|&i| peaks[i]).collect();
        let mut spikes: Vec<TdcSpike> = peaks.iter().map(|&(s, c)| TdcSpike { sample: s, channel: c, unit: 0, amplitude: 0.0 }).collect();
        let delta = self.nbefore.max(self.nafter()) as i64;
        let all_samples: Vec<i64> = spikes.iter().map(|s| s.sample).chain(prev.iter().map(|s| s.sample)).collect();
        let all_chans: Vec<usize> = spikes.iter().map(|s| s.channel).chain(prev.iter().map(|s| s.channel)).collect();
        let ns = spikes.len();
        let neighbours: Vec<Vec<usize>> = (0..ns)
            .map(|i| (0..all_samples.len()).filter(|&j| j != i && (all_samples[j] - all_samples[i]).abs() < delta && self.near[all_chans[i] * m + all_chans[j]]).collect())
            .collect();
        for i in 0..ns {
            let (s, ch) = peaks[i];
            let possible = &self.possible_clusters[ch];
            if possible.is_empty() {
                spikes[i].unit = -1;
                continue;
            }
            let unit = self.most_probable(x, n, possible, s);
            // Best shift of the short template on its channels
            let chans: Vec<usize> = (0..m).filter(|&c| self.templates.sparsity[unit * m + c]).collect();
            let mut best = (0i64, f64::INFINITY);
            for sh in -(self.opts.sample_shift as i64)..=self.opts.sample_shift as i64 {
                let mut d = 0.0f64;
                for &c in &chans {
                    for k in 0..self.width_short {
                        let v = at(x, s - self.nbefore_short as i64 + k as i64 + sh, c) as f64;
                        let t = self.templates.data[(unit * w + self.s0 + k) * m + c] as f64;
                        d += (v - t).powi(2);
                    }
                }
                if d < best.1 {
                    best = (sh, d);
                }
            }
            spikes[i].sample += best.0;
            spikes[i].unit = unit as i64;
            let outer: Vec<usize> = neighbours[i].iter().copied().filter(|&j| j > i && j >= ns).collect();
            let valid = !outer.iter().any(|&j| {
                let b = &prev[j - ns];
                b.sample == spikes[i].sample && b.unit == spikes[i].unit
            });
            if !valid {
                spikes[i].unit = -1;
                continue;
            }
            let inner: Vec<usize> = neighbours[i].iter().copied().filter(|&j| j > i && j < ns).collect();
            for &b in &inner {
                spikes[b].unit = self.most_probable(x, n, possible, spikes[b].sample) as i64;
            }
            let others: Vec<TdcSpike> = inner.iter().map(|&b| spikes[b]).collect();
            let amp = self.fit_amplitude(x, n, &spikes[i], &others);
            let (low, up) = self.opts.amplitude_limits;
            if low <= amp && amp <= up {
                spikes[i].amplitude = amp;
                self.subtract(x, n, &spikes[i]);
            } else if low > amp {
                spikes[i].unit = -1;
            } else {
                spikes[i].amplitude = 1.0;
            }
        }
        spikes.retain(|s| s.unit >= 0);
        spikes
    }

    /// The candidate whose short template is closest to the waveform at `s` (module docs).
    fn most_probable(&self, x: &[f32], n: usize, possible: &[usize], s: i64) -> usize {
        let (w, m) = (self.templates.width, self.templates.channels);
        let union: Vec<usize> = (0..m).filter(|&c| possible.iter().any(|&u| self.templates.sparsity[u * m + c])).collect();
        let mut best = (possible[0], f64::INFINITY);
        for &u in possible {
            let mut d = 0.0f64;
            for &c in &union {
                let has = self.templates.sparsity[u * m + c];
                for k in 0..self.width_short {
                    let v = x[c * n + (s - self.nbefore_short as i64 + k as i64) as usize] as f64;
                    let t = if has { self.templates.data[(u * w + self.s0 + k) * m + c] as f64 } else { 0.0 };
                    d += (v - t).powi(2);
                }
            }
            if d < best.1 {
                best = (u, d);
            }
        }
        best.0
    }

    /// Upstream `fit_one_amplitude_with_neighbors` (least squares, min-norm).
    fn fit_amplitude(&self, x: &[f32], n: usize, spike: &TdcSpike, others: &[TdcSpike]) -> f64 {
        let (w, m) = (self.templates.width, self.templates.channels);
        let u = spike.unit as usize;
        let chans: Vec<usize> = (0..m).filter(|&c| self.templates.sparsity[u * m + c]).collect();
        if chans.is_empty() || self.norms[u] == 0.0 {
            return 0.0;
        }
        let (start, stop) = (spike.sample - self.nbefore as i64, spike.sample + self.nafter() as i64);
        if others.is_empty() {
            let mut dot = 0.0f64;
            for &c in &chans {
                for k in 0..w {
                    dot += self.templates.data[(u * w + k) * m + c] as f64 * x[c * n + (start + k as i64) as usize] as f64;
                }
            }
            return dot / self.norms[u];
        }
        let lim0 = start.min(others.iter().map(|o| o.sample).min().unwrap() - self.nbefore as i64);
        let lim1 = stop.max(others.iter().map(|o| o.sample).max().unwrap() + self.nafter() as i64);
        let len = (lim1 - lim0) as usize;
        // Columns: the spike's template, then each neighbour's (all not fitted yet), on `chans`
        let fitted: Vec<&TdcSpike> = others.iter().filter(|o| o.amplitude == 0.0 && o.unit >= 0).collect();
        let cols = 1 + fitted.len();
        let rows = len * chans.len();
        let mut a = vec![0.0f64; rows * cols];
        let place = |a: &mut [f64], col: usize, sp: &TdcSpike| {
            let unit = sp.unit as usize;
            let i0 = sp.sample - self.nbefore as i64 - lim0;
            for (ci, &c) in chans.iter().enumerate() {
                if self.templates.sparsity[unit * m + c] {
                    for k in 0..w {
                        let r = (i0 + k as i64) as usize;
                        a[(r * chans.len() + ci) * cols + col] += self.templates.data[(unit * w + k) * m + c] as f64;
                    }
                }
            }
        };
        place(&mut a, 0, spike);
        for (j, o) in fitted.iter().enumerate() {
            place(&mut a, j + 1, o);
        }
        let y: Vec<f64> = (0..len).flat_map(|r| chans.iter().map(move |&c| (r, c))).map(|(r, c)| x[c * n + (lim0 + r as i64) as usize] as f64).collect();
        least_squares(&a, &y, rows, cols)[0]
    }

    /// Subtracts `amplitude × template` at the spike.
    fn subtract(&self, x: &mut [f32], n: usize, sp: &TdcSpike) {
        let (w, m) = (self.templates.width, self.templates.channels);
        let u = sp.unit as usize;
        let i0 = sp.sample - self.nbefore as i64;
        for c in (0..m).filter(|&c| self.templates.sparsity[u * m + c]) {
            for k in 0..w {
                x[c * n + (i0 + k as i64) as usize] -= self.templates.data[(u * w + k) * m + c] * sp.amplitude as f32;
            }
        }
    }

    /// `locally_exclusive` on `x[:, lo..hi]` (negative peaks), samples in chunk coordinates.
    fn detect_fast(&self, x: &[f32], n: usize, lo: usize, hi: usize) -> Vec<(i64, usize)> {
        let m = self.templates.channels;
        let len = hi - lo;
        let mut cands = Vec::new();
        for s in 1..len.saturating_sub(1) {
            for c in 0..m {
                let v = x[c * n + lo + s];
                if (v as f64) <= -self.thresholds[c] && v < x[c * n + lo + s - 1] && v <= x[c * n + lo + s + 1] {
                    cands.push(Candidate { sample: s as u64, row: c as u32, score: (v as f64).abs() / self.thresholds[c], tiebreak: 0 });
                }
            }
        }
        let keep = locally_exclusive(&cands, self.sweep as u64, |a, b| self.detect_near[a as usize * m + b as usize], false);
        cands
            .iter()
            .zip(keep)
            .filter(|(c, k)| *k && c.sample as usize >= self.sweep + 1 && (c.sample as usize) < len - self.sweep - 1)
            .map(|(c, _)| ((lo + c.sample as usize) as i64, c.row as usize))
            .collect()
    }

    /// Matched filtering on `x[:, lo..hi]` (module docs), samples in chunk coordinates.
    fn detect_fine(&self, x: &[f32], n: usize, lo: usize, hi: usize) -> Vec<(i64, usize)> {
        let f = self.fine.as_ref().expect("fine detector");
        let m = self.templates.channels;
        let l = f.prototype.len();
        let len = hi - lo;
        if len < 3 * l + 2 {
            return Vec::new();
        }
        // Valid correlation, then cropped by the prototype length on both ends (upstream)
        let valid = len - l + 1;
        let corr: Vec<f32> = (0..m * valid)
            .map(|e| {
                let (c, i) = (e / valid, e % valid);
                (0..l).map(|k| f.prototype[k] * x[c * n + lo + i + k]).sum()
            })
            .collect();
        let p = valid - 2 * l;
        let row = |t: usize, s: usize| -> f32 { f.rows[t].iter().map(|&(c, wt)| wt * corr[c * valid + l + s]).sum() };
        let rows: Vec<Vec<f32>> = (0..m).map(|t| (0..p).map(|s| row(t, s)).collect()).collect();
        let mut cands = Vec::new();
        for s in 1..p.saturating_sub(1) {
            for t in 0..m {
                let v = rows[t][s];
                if (v as f64) >= f.thresholds[t] && v > rows[t][s - 1] && v >= rows[t][s + 1] {
                    cands.push(Candidate { sample: s as u64, row: t as u32, score: v as f64 / f.thresholds[t], tiebreak: 0 });
                }
            }
        }
        let keep = locally_exclusive(&cands, self.sweep as u64, |a, b| self.detect_near[a as usize * m + b as usize], true);
        cands
            .iter()
            .zip(keep)
            .filter(|(c, k)| *k && c.sample as usize >= self.sweep + 1 && (c.sample as usize) < p - self.sweep - 1)
            .map(|(c, _)| ((lo + c.sample as usize + l + f.nbefore) as i64, c.row as usize))
            .collect()
    }
}

/// Minimum-norm least squares `argmin ‖A z − y‖` (`A` row-major `[rows, cols]`), via the
/// eigendecomposition of `AᵀA` with small eigenvalues dropped (LAPACK `gelsd`'s machine-precision
/// cut-off).
fn least_squares(a: &[f64], y: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut g = vec![0.0f64; cols * cols];
    let mut b = vec![0.0f64; cols];
    for r in 0..rows {
        for i in 0..cols {
            let ai = a[r * cols + i];
            if ai != 0.0 {
                b[i] += ai * y[r];
                for j in 0..cols {
                    g[i * cols + j] += ai * a[r * cols + j];
                }
            }
        }
    }
    let eig = symmetric_eigen_cpu(&g, cols);
    let top = eig.values.first().copied().unwrap_or(0.0).max(0.0);
    let cut = (f64::EPSILON * rows.max(cols) as f64).powi(2) * top.max(f64::MIN_POSITIVE);
    let mut z = vec![0.0f64; cols];
    for k in 0..cols {
        let lam = eig.values[k];
        if lam > cut {
            let proj: f64 = (0..cols).map(|i| eig.vectors[i * cols + k] * b[i]).sum::<f64>() / lam;
            for i in 0..cols {
                z[i] += proj * eig.vectors[i * cols + k];
            }
        }
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(s: usize, nb: usize, a: f32) -> f32 {
        let t = s as f32 - nb as f32;
        -a * (-(t / 3.0).powi(2)).exp() + 0.3 * a * (-((t - 8.0) / 5.0).powi(2)).exp()
    }

    /// Two units on separate channels, spikes planted with amplitude 1 (one pair overlapping),
    /// found at their samples with their unit and the residual emptied.
    #[test]
    fn peels_planted_spikes() {
        let fs = 30_000.0;
        let (nb, na, m, n) = (30usize, 75usize, 4usize, 6000usize);
        let w = nb + na;
        let pos: Vec<[f32; 2]> = (0..m).map(|c| [0.0, 40.0 * c as f32]).collect();
        let mut data = vec![0.0f32; 2 * w * m];
        let mut sparsity = vec![false; 2 * m];
        for s in 0..w {
            data[s * m] = wave(s, nb, 12.0);
            data[s * m + 1] = wave(s, nb, 5.0);
            data[w * m + s * m + 2] = wave(s, nb, 10.0);
            data[w * m + s * m + 3] = wave(s, nb, 4.0);
        }
        for (u, c) in [(0usize, 0usize), (0, 1), (1, 2), (1, 3)] {
            sparsity[u * m + c] = true;
        }
        let t = Templates { unit_ids: vec![0, 1], width: w, channels: m, data: data.clone(), sparsity };
        let planted = [(500i64, 0usize), (1500, 1), (1520, 0), (3000, 1), (4500, 0)];
        let mut x = vec![0.0f32; m * n];
        for &(s, u) in &planted {
            for k in 0..w {
                for c in 0..m {
                    x[c * n + (s as usize - nb + k)] += data[(u * w + k) * m + c];
                }
            }
        }
        let peeler = TdcPeeler::new(t, nb, &pos, &[1.0; 4], fs, None, TdcPeelerOptions { use_fine_detector: false, ..Default::default() });
        let spikes = peeler.peel(&mut x, n);
        let got: Vec<(i64, i64)> = spikes.iter().map(|s| (s.sample, s.unit)).collect();
        let want: Vec<(i64, i64)> = planted.iter().map(|&(s, u)| (s, u as i64)).collect();
        assert_eq!(got, want, "{spikes:?}");
        assert!(spikes.iter().all(|s| (s.amplitude - 1.0).abs() < 0.05), "{spikes:?}");
        assert!(x.iter().all(|v| v.abs() < 0.5), "residual left");
    }

    #[test]
    fn least_squares_is_min_norm() {
        // Duplicate columns: the minimum-norm solution splits the coefficient
        let a = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let z = least_squares(&a, &[2.0, 4.0, 6.0], 3, 2);
        assert!((z[0] - 1.0).abs() < 1e-9 && (z[1] - 1.0).abs() < 1e-9, "{z:?}");
    }
}
