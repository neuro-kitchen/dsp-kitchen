//! Learned templates for template matching, from the first clustering's units (Kilosort4 paper,
//! Methods, *Template learning*, written from its description):
//!
//! 1. **Alignment.** Each unit's template is shifted in time so that its best correlation with the
//!    prototype waveforms (`wTEMP`, any channel, any lag) falls at lag 0: templates of one neuron
//!    found in different sections then line up, and spikes matched to them are timed alike.
//! 2. **Merging.** Templates whose correlation, maximized over lags, is at least `min_similarity`
//!    and whose norms differ by less than `max_norm_difference` (relative to their mean) are merged
//!    (spike-count-weighted mean). Largest units first; each pass merges disjoint pairs, then every
//!    similarity is computed again, until a pass merges nothing.
//!
//! Templates stay in PC space (`[units, channels, n_pcs]`), where lagged correlations need no
//! waveforms: with `X[(u·np + p), ch] = a[u, ch, p]`, all pairs' channel products are one
//! `matmul` (`X·Xᵀ`), and combining them with the lagged products of the PCs (`wtw`) gives every
//! pair at every lag (`pair_similarity_kernel`); only the `units²` maxima come back. Shifting a
//! template (alignment) goes through its waveform on the host (a few hundred small templates).
//!
//! Thresholds: Kilosort4's published values (correlation 0.9, norm difference 0.2). Choices of ours:
//! merged templates are averaged without a relative shift (after alignment, templates of one neuron
//! peak at the same lag).

use cubecl::prelude::*;
use dsp_base::core::buffer;
use dsp_base::linalg::{matmul, MatrixView};
use dsp_core::compute::LaunchGeometry;

use super::clustering::SpikeClusters;
use super::kernels::learned::{pair_similarity_kernel, prototype_match_kernel};
use super::matching::lagged_pc_products;
use super::templates::UniversalTemplates;

/// Settings of [`learned_templates`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemplateMergeOptions {
    pub min_similarity: f64,
    pub max_norm_difference: f64,
}

impl Default for TemplateMergeOptions {
    fn default() -> Self {
        Self { min_similarity: 0.9, max_norm_difference: 0.2 }
    }
}

/// Templates for matching: `features[(t · channels + ch) · n_pcs + p]`, waveform `Σ_p features ·
/// wPCA[p]` per channel.
#[derive(Debug, Clone, PartialEq)]
pub struct LearnedTemplates {
    pub n: usize,
    pub channels: usize,
    pub n_pcs: usize,
    pub features: Vec<f32>,
    /// Spikes of the units each template came from.
    pub counts: Vec<usize>,
    /// Units merged away.
    pub merged: usize,
}

/// `proto[p, k, l] = Σ_t wpca[p, t] · wtemp[k, t − lag]`, `lag = l − (nt − 1)`.
fn prototype_products(wpca: &[f32], wtemp: &[f32], np: usize, k: usize, nt: usize) -> Vec<f32> {
    let lags = 2 * nt - 1;
    let mut out = vec![0.0f32; np * k * lags];
    for p in 0..np {
        for kk in 0..k {
            for l in 0..lags {
                let lag = l as isize - (nt as isize - 1);
                out[(p * k + kk) * lags + l] = (0..nt as isize)
                    .filter(|&t| t - lag >= 0 && t - lag < nt as isize)
                    .map(|t| wpca[p * nt + t as usize] as f64 * wtemp[kk * nt + (t - lag) as usize] as f64)
                    .sum::<f64>() as f32;
            }
        }
    }
    out
}

/// Template `a` (`[channels, np]`) shifted by `lag` samples (`W'[t] = W[t + lag]`, 0 outside),
/// through its waveform: back to PC space by projection on the (orthonormal) `wpca` rows.
fn shift_template(a: &mut [f32], lag: isize, wpca: &[f32], np: usize, nt: usize) {
    if lag == 0 {
        return;
    }
    for row in a.chunks_exact_mut(np) {
        if row.iter().all(|&v| v == 0.0) {
            continue;
        }
        let w: Vec<f64> = (0..nt).map(|t| (0..np).map(|p| row[p] as f64 * wpca[p * nt + t] as f64).sum()).collect();
        let shifted = |t: usize| {
            let s = t as isize + lag;
            if s >= 0 && (s as usize) < nt { w[s as usize] } else { 0.0 }
        };
        for (p, v) in row.iter_mut().enumerate() {
            *v = (0..nt).map(|t| shifted(t) * wpca[p * nt + t] as f64).sum::<f64>() as f32;
        }
    }
}

/// Every pair's best lagged inner product (`[units, units]`; the diagonal: squared norms), on the
/// device. Crosses the bus: the templates up, the `units²` maxima down.
fn pair_similarities(client: &Client, features: &[f32], units: usize, channels: usize, np: usize, wtw: &[f32], lags: usize) -> Vec<f32> {
    // X[(u·np + p), ch]
    let rows = units * np;
    let mut x = vec![0.0f32; rows * channels];
    for u in 0..units {
        for ch in 0..channels {
            for p in 0..np {
                x[(u * np + p) * channels + ch] = features[(u * channels + ch) * np + p];
            }
        }
    }
    let x = buffer::upload(client, &x);
    let utu = buffer::empty::<f32>(client, rows * rows);
    let view = MatrixView::row_major(&x, rows * channels, rows, channels);
    matmul::<f32>(client, &view, &view.transposed(), &utu, rows * rows);
    let out = buffer::empty::<f32>(client, units * units);
    let geom = LaunchGeometry::elementwise(client, units * units);
    // SAFETY: `utu` holds `rows²`, `wtw` `np² · lags`, `out` `units²` values
    unsafe {
        pair_similarity_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(utu, rows * rows),
            BufferArg::from_raw_parts(buffer::upload(client, wtw), wtw.len()),
            BufferArg::from_raw_parts(out.clone(), units * units),
            units as u32,
            np as u32,
            lags as u32,
        );
    }
    buffer::download_prefix::<f32>(client, out, units * units)
}

/// Aligned, merged templates of the units of `clusters` (module docs).
pub fn learned_templates(client: &Client, clusters: &SpikeClusters, universal: &UniversalTemplates, opts: &TemplateMergeOptions) -> LearnedTemplates {
    let (units, channels, np, nt, k) = (clusters.n_units, clusters.channels, clusters.n_pcs, universal.nt, universal.n_templates);
    let lags = 2 * nt - 1;
    let mut counts = vec![0usize; units];
    clusters.labels.iter().for_each(|&u| counts[u as usize] += 1);
    let mut features = clusters.templates.clone();
    if units == 0 {
        return LearnedTemplates { n: 0, channels, n_pcs: np, features, counts, merged: 0 };
    }

    // 1. Alignment: best (prototype, lag) on any channel, on the device
    let proto = prototype_products(&universal.wpca, &universal.wtemp, np, k, nt);
    let scores = buffer::empty::<f32>(client, units * k * lags);
    let geom = LaunchGeometry::elementwise(client, units * k * lags);
    // SAFETY: `features` holds `units · channels · np`, `proto` `np · k · lags`, `scores` `units · k · lags`
    unsafe {
        prototype_match_kernel::launch::<f32>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(buffer::upload(client, &features), features.len()),
            BufferArg::from_raw_parts(buffer::upload(client, &proto), proto.len()),
            BufferArg::from_raw_parts(scores.clone(), units * k * lags),
            units as u32,
            channels as u32,
            np as u32,
            k as u32,
            lags as u32,
        );
    }
    let scores = buffer::download_prefix::<f32>(client, scores, units * k * lags);
    for u in 0..units {
        let s = &scores[u * k * lags..(u + 1) * k * lags];
        let best = (0..s.len()).fold(0, |b, i| if s[i] > s[b] { i } else { b });
        let lag = (best % lags) as isize - (nt as isize - 1);
        shift_template(&mut features[u * channels * np..(u + 1) * channels * np], lag, &universal.wpca, np, nt);
    }

    // 2. Merging passes
    let wtw = lagged_pc_products(&universal.wpca, np, nt);
    let mut alive: Vec<usize> = (0..units).filter(|&u| counts[u] > 0).collect();
    let mut merged = 0;
    loop {
        let n = alive.len();
        let packed: Vec<f32> = alive.iter().flat_map(|&u| features[u * channels * np..(u + 1) * channels * np].iter().copied()).collect();
        let sim = pair_similarities(client, &packed, n, channels, np, &wtw, lags);
        let norm: Vec<f64> = (0..n).map(|a| (sim[a * n + a].max(0.0) as f64).sqrt()).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| counts[alive[b]].cmp(&counts[alive[a]]).then(a.cmp(&b)));
        let (mut touched, mut gone) = (vec![false; n], vec![false; n]);
        let mut merges = 0;
        for &a in &order {
            if touched[a] || norm[a] == 0.0 {
                continue;
            }
            let mut partners: Vec<(f64, usize)> = (0..n)
                .filter(|&b| b != a && !touched[b] && norm[b] > 0.0)
                .map(|b| (sim[a * n + b] as f64 / (norm[a] * norm[b]), b))
                .collect();
            partners.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
            for (r, b) in partners {
                if r < opts.min_similarity {
                    break;
                }
                if (2.0 * (norm[a] - norm[b]) / (norm[a] + norm[b])).abs() < opts.max_norm_difference {
                    let (ua, ub) = (alive[a], alive[b]);
                    let (ca, cb) = (counts[ua] as f32, counts[ub] as f32);
                    for i in 0..channels * np {
                        let v = (ca * features[ua * channels * np + i] + cb * features[ub * channels * np + i]) / (ca + cb);
                        features[ua * channels * np + i] = v;
                    }
                    counts[ua] += counts[ub];
                    counts[ub] = 0;
                    touched[a] = true;
                    touched[b] = true;
                    gone[b] = true;
                    merges += 1;
                    break;
                }
            }
        }
        merged += merges;
        alive = alive.into_iter().zip(gone).filter(|&(_, g)| !g).map(|(u, _)| u).collect();
        if merges == 0 {
            break;
        }
    }
    let out: Vec<f32> = alive.iter().flat_map(|&u| features[u * channels * np..(u + 1) * channels * np].iter().copied()).collect();
    LearnedTemplates { n: alive.len(), channels, n_pcs: np, features: out, counts: alive.iter().map(|&u| counts[u]).collect(), merged }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Orthonormal "PCs": unit impulses at a few samples (so waveforms and features coincide there).
    fn impulse_pcs(np: usize, nt: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; np * nt];
        for p in 0..np {
            w[p * nt + nt / 2 - np / 2 + p] = 1.0;
        }
        w
    }

    #[test]
    fn lagged_products_of_orthonormal_pcs() {
        let (np, nt) = (3, 9);
        let wpca = impulse_pcs(np, nt);
        let wtw = lagged_pc_products(&wpca, np, nt);
        let lags = 2 * nt - 1;
        let at = |p: usize, q: usize, lag: isize| wtw[(p * np + q) * lags + (lag + nt as isize - 1) as usize];
        // Impulses at consecutive samples: pc 1 sits one sample after pc 0, so they meet at lag +1
        assert_eq!(at(0, 0, 0), 1.0);
        assert_eq!(at(0, 1, 1), 1.0);
        assert_eq!(at(0, 1, -1), 0.0);
        assert_eq!(at(1, 1, 0), 1.0);
    }

    #[test]
    fn shift_moves_the_waveform() {
        // Full basis (np = nt): shifting is exact
        let (np, nt) = (5, 5);
        let mut wpca = vec![0.0f32; np * nt];
        (0..np).for_each(|p| wpca[p * nt + p] = 1.0);
        let mut a = vec![0.0, 1.0, 2.0, 0.0, 0.0];
        shift_template(&mut a, 1, &wpca, np, nt);
        assert_eq!(a, vec![1.0, 2.0, 0.0, 0.0, 0.0], "W'[t] = W[t + 1]");
    }

    /// On the device: a copy of a template (scaled within the norm tolerance) merges into it, a
    /// different one does not; counts add up.
    #[test]
    fn near_copies_merge_and_others_do_not() {
        let Ok(target) = dsp_core::compute::ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (channels, np, nt, k) = (4usize, 3usize, 9usize, 2usize);
        let wpca = impulse_pcs(np, nt);
        // Prototypes: the PCs' first two impulses (alignment keeps the templates in place)
        let wtemp: Vec<f32> = wpca[..k * nt].to_vec();
        let universal = UniversalTemplates { nt, n_pcs: np, n_templates: k, wpca, wtemp };
        let a = [1.0f32, 0.2, 0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let copy: Vec<f32> = a.iter().map(|v| v * 1.1).collect();
        let other = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 1.0, 0.0, 0.9, 0.0];
        let templates: Vec<f32> = a.iter().chain(&copy).chain(&other).copied().collect();
        let labels: Vec<u32> = [vec![0u32; 50], vec![1u32; 30], vec![2u32; 20]].concat();
        let clusters = SpikeClusters { labels, n_units: 3, templates, channels, n_pcs: np, sections: 1 };
        let learned = learned_templates(&client, &clusters, &universal, &TemplateMergeOptions::default());
        assert_eq!(learned.n, 2, "{learned:?}");
        assert_eq!(learned.merged, 1);
        let mut counts = learned.counts.clone();
        counts.sort_unstable();
        assert_eq!(counts, vec![20, 80]);
    }
}
