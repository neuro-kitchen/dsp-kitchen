//! SpikeInterface's cluster templates and their cleaning and merging (`sortingcomponents/clustering/
//! tools.py`, `merging_tools.py`, `sortingcomponents/tools.py` `clean_templates`,
//! `postprocessing/template_similarity.py`; MIT), as SpyKING CIRCUS 2 runs them.
//!
//! - [`templates_from_svd`]: per unit, the peaks on its most frequent channel; per channel of that
//!   channel's sparse set, the median (or mean) of each SVD component, mapped back through the components
//!   (`TruncatedSVD.inverse_transform`: no mean). Also each channel's largest standard deviation
//!   over time of the units' reconstructed waveforms.
//! - [`clean_templates`]: sparsify to channels with peak-to-peak / noise ≥ `sparsify_threshold`;
//!   drop empty templates, templates whose trough on their main channel is more than `max_jitter`
//!   samples from `nbefore`, templates with no channel of peak-to-peak / noise ≥ `min_snr`, and
//!   templates whose mean of max-std / noise over their channels exceeds `mean_sd_ratio_threshold`.
//! - [`template_similarity`]: `1 − Σ|a − b| / (Σ|a| + Σ|b|)` over the union of two templates'
//!   channels (pairs sharing none: similarity 0), the best of `±num_shifts` lags.
//! - [`merge_by_similarity`]: pairs above `similarity_thresh` joined into connected components;
//!   each component keeps its first unit's id, its template the count-weighted average of the
//!   members' templates shifted by their lag to the first, its channels those all members share.
//!
//! Upstream quirks kept as they are (we validate against upstream): a template array compared with
//! itself fills both orientations of a pair, and both shift signs, with the value at the negative
//! shift ([`template_similarity`]), so only a later unit shifted earlier is aligned and every lag is
//! ≤ 0; the merge then shifts a member by `lags[member, first]` as if lags were antisymmetric,
//! which moves it the wrong way (by twice the lag) when it is not 0.
//! - [`remove_small_clusters`]: units with fewer than `int(duration · min_firing_rate /
//!   subsampling_factor)` peaks dropped.

use std::collections::BTreeMap;

use dsp_synapse::features::ChannelNeighbourhoods;

/// Dense templates `[units, width, channels]` with their unit ids and sparsity.
#[derive(Debug, Clone, PartialEq)]
pub struct Templates {
    pub unit_ids: Vec<i64>,
    pub width: usize,
    pub channels: usize,
    /// `[units, width, channels]`, zeros off the sparsity.
    pub data: Vec<f32>,
    /// `[units, channels]`.
    pub sparsity: Vec<bool>,
}

impl Templates {
    pub fn len(&self) -> usize {
        self.unit_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.unit_ids.is_empty()
    }

    fn unit(&self, u: usize) -> &[f32] {
        &self.data[u * self.width * self.channels..(u + 1) * self.width * self.channels]
    }

    /// Only the units at `keep` (in order).
    pub fn select(&self, keep: &[usize]) -> Self {
        let (w, m) = (self.width, self.channels);
        Self {
            unit_ids: keep.iter().map(|&u| self.unit_ids[u]).collect(),
            width: w,
            channels: m,
            data: keep.iter().flat_map(|&u| self.unit(u).iter().copied()).collect(),
            sparsity: keep.iter().flat_map(|&u| self.sparsity[u * m..(u + 1) * m].iter().copied()).collect(),
        }
    }
}

/// NumPy's median (`v` reordered).
fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    let (lower, mid, _) = v.select_nth_unstable_by(n / 2, f64::total_cmp);
    let mid = *mid;
    if n % 2 == 1 { mid } else { 0.5 * (lower.iter().copied().fold(f64::NEG_INFINITY, f64::max) + mid) }
}

/// Templates of the labelled peaks (labels ≥ 0) from their sparse SVD features (module docs).
/// `features`: `[peaks, components, max_neighbours]` on `mask`'s rows; `components`: `[components,
/// width]`. Returns the templates (sparsity: the best channel's sparse set) and `max_std`
/// (`[units, channels]`).
pub fn templates_from_svd(
    labels: &[i64],
    peak_channels: &[u32],
    features: &[f32],
    mask: &ChannelNeighbourhoods,
    components: &[f32],
    n_components: usize,
    width: usize,
    channels: usize,
    median_operator: bool,
) -> (Templates, Vec<f32>) {
    let nb = mask.max_neighbours;
    let mut by_label: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if l > -1 {
            by_label.entry(l).or_default().push(i);
        }
    }
    let k = by_label.len();
    let mut data = vec![0.0f32; k * width * channels];
    let mut sparsity = vec![false; k * channels];
    let mut max_std = vec![0.0f32; k * channels];
    let mut col = Vec::new();
    for (u, (_, peaks)) in by_label.iter().enumerate() {
        // Most frequent channel; ties to the lowest (np.unique sorts, argmax takes the first)
        let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
        for &i in peaks {
            *counts.entry(peak_channels[i]).or_default() += 1;
        }
        let best = counts.iter().fold((0u32, 0usize), |b, (&c, &n)| if n > b.1 { (c, n) } else { b }).0 as usize;
        let sub: Vec<usize> = peaks.iter().copied().filter(|&i| peak_channels[i] as usize == best).collect();
        for (slot, ch) in mask.of(best).enumerate() {
            sparsity[u * channels + ch] = true;
            let meds: Vec<f64> = (0..n_components)
                .map(|c| {
                    col.clear();
                    col.extend(sub.iter().map(|&i| features[(i * n_components + c) * nb + slot] as f64));
                    if median_operator { median(&mut col) } else { col.iter().sum::<f64>() / col.len() as f64 }
                })
                .collect();
            for t in 0..width {
                let v: f64 = (0..n_components).map(|c| meds[c] * components[c * width + t] as f64).sum();
                data[(u * width + t) * channels + ch] = v as f32;
            }
            if sub.len() > 1 {
                // Each peak's waveform, then the standard deviation over peaks at every sample
                let n = sub.len() as f64;
                let mut best_std = 0.0f64;
                for t in 0..width {
                    let vals: Vec<f64> = sub
                        .iter()
                        .map(|&i| (0..n_components).map(|c| features[(i * n_components + c) * nb + slot] as f64 * components[c * width + t] as f64).sum())
                        .collect();
                    let mean = vals.iter().sum::<f64>() / n;
                    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                    best_std = best_std.max(var.sqrt());
                }
                max_std[u * channels + ch] = best_std as f32;
            }
        }
    }
    (Templates { unit_ids: by_label.keys().copied().collect(), width, channels, data, sparsity }, max_std)
}

/// Settings of [`clean_templates`] (SpyKING CIRCUS 2's in [`Default`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CleanOptions {
    pub sparsify_threshold: Option<f64>,
    pub min_snr: Option<f64>,
    /// Samples.
    pub max_jitter: Option<usize>,
    pub remove_empty: bool,
    pub mean_sd_ratio_threshold: f64,
}

impl Default for CleanOptions {
    fn default() -> Self {
        Self { sparsify_threshold: Some(1.0), min_snr: Some(5.0), max_jitter: Some(6), remove_empty: true, mean_sd_ratio_threshold: 3.0 }
    }
}

fn peak_to_peak(t: &[f32], width: usize, channels: usize, ch: usize) -> f64 {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for s in 0..width {
        let v = t[s * channels + ch];
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (hi - lo) as f64
}

/// The cleaned templates (module docs) and the index (in the input) of each kept unit.
/// `max_std`: `[units, channels]` of [`templates_from_svd`] (`None` skips that test).
pub fn clean_templates(templates: &Templates, noise: &[f64], nbefore: usize, max_std: Option<&[f32]>, opts: &CleanOptions) -> (Templates, Vec<usize>) {
    let (w, m) = (templates.width, templates.channels);
    let mut t = templates.clone();
    if let Some(thr) = opts.sparsify_threshold {
        for u in 0..t.len() {
            for ch in 0..m {
                let keep = peak_to_peak(templates.unit(u), w, m, ch) / noise[ch] >= thr;
                t.sparsity[u * m + ch] = keep;
                if !keep {
                    for s in 0..w {
                        t.data[(u * w + s) * m + ch] = 0.0;
                    }
                }
            }
        }
    }
    let mut keep: Vec<usize> = (0..t.len()).collect();
    if opts.remove_empty {
        keep.retain(|&u| t.unit(u).iter().any(|&v| v != 0.0));
    }
    if let Some(jitter) = opts.max_jitter {
        keep.retain(|&u| {
            let tp = t.unit(u);
            // Main channel: the deepest trough; its trough sample
            let (mut best, mut ch) = (f32::INFINITY, 0usize);
            for c in 0..m {
                for s in 0..w {
                    if tp[s * m + c] < best {
                        best = tp[s * m + c];
                        ch = c;
                    }
                }
            }
            let trough = (0..w).fold(0usize, |b, s| if tp[s * m + ch] < tp[b * m + ch] { s } else { b });
            (trough as i64 - nbefore as i64).unsigned_abs() as usize <= jitter
        });
    }
    if let Some(snr) = opts.min_snr {
        keep.retain(|&u| (0..m).any(|ch| peak_to_peak(t.unit(u), w, m, ch) / noise[ch] >= snr));
    }
    if let Some(std) = max_std {
        keep.retain(|&u| {
            let chans: Vec<usize> = (0..m).filter(|&c| t.sparsity[u * m + c]).collect();
            if chans.is_empty() {
                return false;
            }
            let ratio = chans.iter().map(|&c| std[u * m + c] as f64 / noise[c]).sum::<f64>() / chans.len() as f64;
            ratio <= opts.mean_sd_ratio_threshold
        });
    }
    (t.select(&keep), keep)
}

/// `(similarity, lags)`, both `[units, units]` (module docs; `support="union"`, method l1).
///
/// As upstream compares an array of templates with itself: only the pairs `i ≤ j` at shifts
/// `−num_shifts..=0` are computed (`D(i, j, s)`: unit `i` on `[n, w − n)` against unit `j` shifted by
/// `s`); every orientation and sign takes that pair's value (`d(i, j, ±s) = d(j, i, ±s) = D(min,
/// max, −s)`), so the lag (the first smallest distance) is never positive.
pub fn template_similarity(t: &Templates, num_shifts: usize) -> (Vec<f64>, Vec<i32>) {
    let (k, w, m) = (t.len(), t.width, t.channels);
    let span = w - 2 * num_shifts;
    let ns = num_shifts as i64;
    // D(i, j, s) for i ≤ j, s = −ns..=0 (index s + ns); 1 when the pair shares no channel
    let mut d = vec![1.0f64; k * k * (num_shifts + 1)];
    for i in 0..k {
        for j in i..k {
            let si = &t.sparsity[i * m..(i + 1) * m];
            let sj = &t.sparsity[j * m..(j + 1) * m];
            if !(0..m).any(|c| si[c] && sj[c]) {
                continue;
            }
            let chans: Vec<usize> = (0..m).filter(|&c| si[c] || sj[c]).collect();
            let (a, b) = (t.unit(i), t.unit(j));
            for shift in -ns..=0 {
                let (mut diff, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
                for s in 0..span {
                    let (sa, sb) = (s + num_shifts, (s as i64 + ns + shift) as usize);
                    for &c in &chans {
                        let (x, y) = (a[sa * m + c] as f64, b[sb * m + c] as f64);
                        diff += (x - y).abs();
                        na += x.abs();
                        nb += y.abs();
                    }
                }
                d[(i * k + j) * (num_shifts + 1) + (shift + ns) as usize] = diff / (na + nb);
            }
        }
    }
    let mut sim = vec![0.0f64; k * k];
    let mut lag = vec![0i32; k * k];
    for i in 0..k {
        for j in 0..k {
            let (lo, hi) = (i.min(j), i.max(j));
            let at = |shift: i64| d[(lo * k + hi) * (num_shifts + 1) + (ns - shift.abs()) as usize];
            // Shifts in upstream's order −n..=n; argmin keeps the first of equal distances
            let (mut best, mut best_lag) = (f64::INFINITY, 0i64);
            for shift in -ns..=ns {
                let v = at(shift);
                if v < best {
                    best = v;
                    best_lag = shift;
                }
            }
            sim[i * k + j] = 1.0 - best;
            lag[i * k + j] = best_lag as i32;
        }
    }
    (sim, lag)
}

/// Result of [`merge_by_similarity`].
#[derive(Debug, Clone, PartialEq)]
pub struct Merged {
    pub labels: Vec<i64>,
    pub templates: Templates,
    /// Per peak: the lag added for the units merged into another (samples).
    pub time_shifts: Vec<i32>,
}

/// Merges units whose templates are alike (module docs).
pub fn merge_by_similarity(labels: &[i64], t: &Templates, similarity_thresh: f64, num_shifts: usize, use_lags: bool) -> Merged {
    let (k, w, m) = (t.len(), t.width, t.channels);
    let (sim, lags) = template_similarity(t, num_shifts);
    // Connected components, numbered in order of their lowest unit (scipy's labelling)
    let mut comp = vec![usize::MAX; k];
    let mut n_comp = 0;
    for s in 0..k {
        if comp[s] != usize::MAX {
            continue;
        }
        let mut stack = vec![s];
        comp[s] = n_comp;
        while let Some(a) = stack.pop() {
            for b in 0..k {
                if comp[b] == usize::MAX && (sim[a * k + b] > similarity_thresh || sim[b * k + a] > similarity_thresh) {
                    comp[b] = n_comp;
                    stack.push(b);
                }
            }
        }
        n_comp += 1;
    }
    let mut labels = labels.to_vec();
    let mut time_shifts = vec![0i32; labels.len()];
    let mut data = t.data.clone();
    let mut sparsity = t.sparsity.clone();
    let mut keep = vec![true; k];
    let mut unit_ids = Vec::with_capacity(n_comp);
    for c in 0..n_comp {
        let group: Vec<usize> = (0..k).filter(|&u| comp[u] == c).collect();
        let g0 = group[0];
        unit_ids.push(t.unit_ids[g0]);
        if group.len() < 2 {
            continue;
        }
        let mut weights = vec![0.0f64; group.len()];
        for (gi, &l) in group.iter().enumerate() {
            let id = t.unit_ids[l];
            let members: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] == id).collect();
            weights[gi] = members.len() as f64;
            if gi > 0 {
                keep[l] = false;
                for &i in &members {
                    labels[i] = t.unit_ids[g0];
                    if use_lags {
                        time_shifts[i] += lags[l * k + g0];
                    }
                }
            }
        }
        let total: f64 = weights.iter().sum();
        let mut acc = vec![0.0f64; w * m];
        for (gi, &l) in group.iter().enumerate() {
            let shift = if use_lags { lags[l * k + g0] as i64 } else { 0 };
            let src = t.unit(l);
            for s in 0..w {
                let from = s as i64 - shift;
                if (0..w as i64).contains(&from) {
                    for ch in 0..m {
                        acc[s * m + ch] += src[from as usize * m + ch] as f64 * weights[gi] / total;
                    }
                }
            }
        }
        data[g0 * w * m..(g0 + 1) * w * m].iter_mut().zip(&acc).for_each(|(d, a)| *d = *a as f32);
        for ch in 0..m {
            sparsity[g0 * m + ch] = group.iter().all(|&l| t.sparsity[l * m + ch]);
        }
    }
    let kept: Vec<usize> = (0..k).filter(|&u| keep[u]).collect();
    let merged = Templates { unit_ids: t.unit_ids.clone(), width: w, channels: m, data, sparsity }.select(&kept);
    Merged { labels, templates: Templates { unit_ids, ..merged }, time_shifts }
}

/// Labels with the units of fewer than `int(duration_sec · min_firing_rate / subsampling_factor)`
/// peaks set to −1, and the unit ids kept (sorted).
pub fn remove_small_clusters(labels: &[i64], duration_sec: f64, min_firing_rate: f64, subsampling_factor: f64) -> (Vec<i64>, Vec<i64>) {
    let min_count = (duration_sec * min_firing_rate / subsampling_factor) as usize;
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for &l in labels {
        *counts.entry(l).or_default() += 1;
    }
    let out = labels.iter().map(|&l| if counts[&l] < min_count { -1 } else { l }).collect();
    let kept = counts.iter().filter(|(l, n)| **n >= min_count && **l >= 0).map(|(&l, _)| l).collect();
    (out, kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(shift: f32, amp: f32, w: usize) -> Vec<f32> {
        (0..w).map(|s| -amp * (-((s as f32 - 10.0 - shift) / 2.0).powi(2)).exp()).collect()
    }

    fn templates(waves: &[(Vec<f32>, usize)], w: usize, m: usize) -> Templates {
        let mut data = vec![0.0f32; waves.len() * w * m];
        let mut sparsity = vec![false; waves.len() * m];
        for (u, (wv, ch)) in waves.iter().enumerate() {
            for s in 0..w {
                data[(u * w + s) * m + ch] = wv[s];
            }
            sparsity[u * m + ch] = true;
        }
        Templates { unit_ids: (0..waves.len() as i64).collect(), width: w, channels: m, data, sparsity }
    }

    #[test]
    fn similarity_finds_the_lag_and_merges() {
        let (w, m) = (30, 2);
        // Unit 1 is unit 0 one sample earlier (the direction upstream's fill can align); unit 2 is
        // on another channel
        let t = templates(&[(wave(0.0, 10.0, w), 0), (wave(-1.0, 10.0, w), 0), (wave(0.0, 10.0, w), 1)], w, m);
        let (later, _) = template_similarity(&templates(&[(wave(0.0, 10.0, w), 0), (wave(1.0, 10.0, w), 0)], w, m), 3);
        assert!(later[1] < 0.8, "one sample later is not aligned (upstream): {}", later[1]);
        let (sim, lag) = template_similarity(&t, 3);
        assert!(sim[0 * 3 + 1] > 0.95, "{}", sim[1]);
        // Upstream's symmetric fill: both orientations take the pair's value at −1
        assert_eq!(lag[0 * 3 + 1], -1);
        assert_eq!(lag[1 * 3 + 0], -1);
        assert_eq!(sim[0 * 3 + 2], 0.0, "no shared channel");
        let labels = vec![0, 0, 0, 1, 2, 2];
        let merged = merge_by_similarity(&labels, &t, 0.8, 3, true);
        assert_eq!(merged.labels, vec![0, 0, 0, 0, 2, 2]);
        assert_eq!(merged.templates.unit_ids, vec![0, 2]);
        assert_eq!(merged.time_shifts[3], -1, "unit 1's peak moved by its lag to unit 0");
    }

    #[test]
    fn cleaning_and_small_clusters() {
        let (w, m) = (30, 2);
        let t = templates(&[(wave(0.0, 10.0, w), 0), (wave(5.0, 10.0, w), 0), (wave(0.0, 2.0, w), 1)], w, m);
        let (clean, kept) = clean_templates(&t, &[1.0, 1.0], 10, None, &CleanOptions { max_jitter: Some(2), ..Default::default() });
        assert_eq!(kept, vec![0], "unit 1 is off-centre, unit 2 below SNR 5");
        assert_eq!(clean.unit_ids, vec![0]);
        let (labels, kept) = remove_small_clusters(&[0, 0, 0, 1, -1], 100.0, 0.02, 1.0);
        assert_eq!(labels, vec![0, 0, 0, -1, -1]);
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn svd_templates_are_medians_mapped_back() {
        let pos: Vec<[f32; 2]> = (0..2).map(|c| [0.0, 20.0 * c as f32]).collect();
        let mask = ChannelNeighbourhoods::within_radius(&pos, 25.0);
        let nb = mask.max_neighbours;
        // One component equal to the unit waveform; features 1, 2, 9 on slot 0 → median 2
        let comp = vec![1.0f32, -2.0, 1.0];
        let mut feats = vec![0.0f32; 3 * nb];
        for (i, v) in [1.0f32, 2.0, 9.0].iter().enumerate() {
            feats[i * nb] = *v;
        }
        let (t, std) = templates_from_svd(&[5, 5, 5], &[0, 0, 0], &feats, &mask, &comp, 1, 3, 2, true);
        assert_eq!(t.unit_ids, vec![5]);
        assert_eq!(&t.data[..], &[2.0, 0.0, -4.0, 0.0, 2.0, 0.0]);
        assert!(std[0] > 0.0);
    }
}
