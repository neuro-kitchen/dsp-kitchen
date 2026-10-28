//! SpikeInterface's `auto_merge_units` with the `x_contaminations` preset (`curation/auto_merge.py`,
//! `curation/curation_tools.py`, `core/sorting_tools.py`, `metrics/quality/misc_metrics.py`; MIT), as
//! SpyKING CIRCUS 2's `final_cleaning_circus` runs it: one pass per template-difference threshold
//! (`template_diff_thresh`), each repeated while it merges something.
//!
//! One iteration, over the pairs `i < j`:
//! 1. units with fewer than `min_spikes` spikes, or a refractory contamination (Llobet & Wyngaard
//!    2022, `1 − √(1 − n_v (T − 2 N t_c) / (N² (t_r − t_c)))`) above `contamination_thresh`, take
//!    part in no pair;
//! 2. unit locations (template peak-to-peak centre of mass over the unit's channels) at most
//!    `max_distance_um` apart;
//! 3. template difference (`1 − similarity`, [`super::template_similarity`]) below the threshold;
//! 4. cross-contamination test: the coincidences of the two trains within the refractory period
//!    (minus those within the censored period, both with upstream's fractional borders) against a
//!    binomial with `n = N₁ N₂ ((1 − C₁) cc_thresh + C₁)`, `p = 2 t_r / T`; pairs whose p-value
//!    exceeds `p_value` stay;
//! 5. the merge must not lower `firing rate · (1 − (1 + balance) · contamination)` of either unit
//!    (merged train censored at `censored_period_ms`).
//!
//! The pairs are joined into connected components; a group merges when its units' channels overlap
//! (intersection / union) by at least `sparsity_overlap`. A merged unit gets the next free id; its
//! spikes are the group's, every spike closer than `censor_ms` to the previous one dropped; its
//! template the spike-count-weighted average on the channels all members share; its location and
//! its similarities to every unit computed anew (full ±shifts, the new template against all).
//!
//! The binomial survival function is SciPy's `binom.sf` evaluated by `interp1d(kind="quadratic")`
//! over the integers around `n` (upstream's `binom_sf`): the regularised incomplete beta function
//! (continued fraction) and a degree-2 interpolating B-spline with SciPy's knots.

use std::collections::BTreeMap;

use super::templates::Templates;

/// Settings of [`auto_merge`] (SpyKING CIRCUS 2's in [`Default`]).
#[derive(Debug, Clone, PartialEq)]
pub struct AutoMergeOptions {
    pub template_diff_thresholds: Vec<f64>,
    pub min_spikes: usize,
    pub contamination_thresh: f64,
    pub refractory_period_ms: f64,
    pub censored_period_ms: f64,
    pub max_distance_um: f64,
    pub cc_thresh: f64,
    pub p_value: f64,
    pub firing_contamination_balance: f64,
    pub sparsity_overlap: f64,
    pub censor_ms: f64,
    /// Template similarity lags (samples) each side.
    pub num_shifts: usize,
}

impl Default for AutoMergeOptions {
    fn default() -> Self {
        Self {
            template_diff_thresholds: (1..10).map(|i| 0.05 * i as f64).collect(),
            min_spikes: 100,
            contamination_thresh: 0.2,
            refractory_period_ms: 1.0,
            censored_period_ms: 0.3,
            max_distance_um: 50.0,
            cc_thresh: 0.1,
            p_value: 0.2,
            firing_contamination_balance: 1.5,
            sparsity_overlap: 0.5,
            censor_ms: 3.0,
            num_shifts: 3,
        }
    }
}

/// The units after merging: each unit's spike indices (into the input spikes, sorted by sample) and
/// templates (ids as upstream numbers them).
#[derive(Debug, Clone)]
pub struct AutoMerged {
    pub spikes: Vec<Vec<usize>>,
    pub templates: Templates,
    /// Merges applied (groups of the ids at the time).
    pub merges: usize,
}

/// Units' spikes (`units[u]`: indices into `samples`, sorted by sample) with their templates, merged
/// (module docs). `samples`: every spike's sample; `positions`: `(x, y)` µm per channel.
pub fn auto_merge(samples: &[u64], units: Vec<Vec<usize>>, templates: Templates, positions: &[[f32; 2]], fs: f64, total: u64, opts: &AutoMergeOptions) -> AutoMerged {
    let mut units = units;
    let mut t = templates;
    let mut sim = super::templates::template_similarity(&t, opts.num_shifts).0;
    let mut loc: Vec<[f64; 2]> = (0..t.len()).map(|u| centre_of_mass(&t, u, positions)).collect();
    let mut merges = 0;
    for &thresh in &opts.template_diff_thresholds {
        loop {
            let before = t.len();
            let groups = merge_groups(samples, &units, &t, &sim, &loc, fs, total, thresh, opts);
            if groups.is_empty() {
                break;
            }
            let m = t.channels;
            let groups: Vec<Vec<usize>> = groups
                .into_iter()
                .filter(|g| {
                    let union = (0..m).filter(|&c| g.iter().any(|&u| t.sparsity[u * m + c])).count();
                    let inter = (0..m).filter(|&c| g.iter().all(|&u| t.sparsity[u * m + c])).count();
                    union > 0 && inter as f64 / union as f64 >= opts.sparsity_overlap
                })
                .collect();
            if groups.is_empty() {
                break;
            }
            apply_merges(samples, &mut units, &mut t, &mut sim, &mut loc, &groups, positions, fs, opts);
            merges += groups.len();
            if t.len() >= before {
                break;
            }
        }
    }
    AutoMerged { spikes: units, templates: t, merges }
}

/// One iteration's merge groups (unit indices; module docs, steps 1–5 and the graph).
#[allow(clippy::too_many_arguments)]
fn merge_groups(samples: &[u64], units: &[Vec<usize>], t: &Templates, sim: &[f64], loc: &[[f64; 2]], fs: f64, total: u64, thresh: f64, opts: &AutoMergeOptions) -> Vec<Vec<usize>> {
    let k = t.len();
    let trains: Vec<Vec<i64>> = units.iter().map(|u| u.iter().map(|&i| samples[i] as i64).collect()).collect();
    let t_c = (opts.censored_period_ms * fs * 1e-3).round();
    let t_r = (opts.refractory_period_ms * fs * 1e-3).round();
    let cont: Vec<f64> = trains.iter().map(|tr| rp_contamination(tr, total as f64, t_c, t_r)).collect();
    let duration = total as f64 / fs;
    let mut pairs = Vec::new();
    for i in 0..k {
        if trains[i].len() < opts.min_spikes || cont[i] > opts.contamination_thresh {
            continue;
        }
        for j in i + 1..k {
            if trains[j].len() < opts.min_spikes || cont[j] > opts.contamination_thresh {
                continue;
            }
            let d = (loc[i][0] - loc[j][0]).hypot(loc[i][1] - loc[j][1]);
            if d > opts.max_distance_um || 1.0 - sim[i * k + j] >= thresh {
                continue;
            }
            let p = cross_contamination_p(&trains[i], &trains[j], fs, total as f64, opts, cont[i]);
            if p <= opts.p_value {
                continue;
            }
            // Quality score of the merged train (censored at the censored period)
            let delta = (opts.censored_period_ms / 1000.0 * fs) as i64;
            let merged = non_duplicated(&trains[i], &trains[j], delta);
            let c_new = rp_contamination(&merged, total as f64, t_c, t_r);
            let (f1, f2, fnew) = (trains[i].len() as f64 / duration, trains[j].len() as f64 / duration, merged.len() as f64 / duration);
            let kk = 1.0 + opts.firing_contamination_balance;
            let (s1, s2, snew) = (f1 * (1.0 - kk * cont[i]), f2 * (1.0 - kk * cont[j]), fnew * (1.0 - kk * c_new));
            if snew < s1 || snew < s2 {
                continue;
            }
            pairs.push((i, j));
        }
    }
    // Connected components with more than one unit, units in index order
    let mut parent: Vec<usize> = (0..k).collect();
    fn find(p: &mut [usize], mut a: usize) -> usize {
        while p[a] != a {
            p[a] = p[p[a]];
            a = p[a];
        }
        a
    }
    for &(a, b) in &pairs {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    let mut comps: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for u in 0..k {
        let r = find(&mut parent, u);
        comps.entry(r).or_default().push(u);
    }
    comps.into_values().filter(|g| g.len() > 1).collect()
}

/// Applies the merge groups (module docs).
#[allow(clippy::too_many_arguments)]
fn apply_merges(samples: &[u64], units: &mut Vec<Vec<usize>>, t: &mut Templates, sim: &mut Vec<f64>, loc: &mut Vec<[f64; 2]>, groups: &[Vec<usize>], positions: &[[f32; 2]], fs: f64, opts: &AutoMergeOptions) {
    let (k, w, m) = (t.len(), t.width, t.channels);
    let rpv = (fs * opts.censor_ms / 1000.0) as u64;
    let next_id = t.unit_ids.iter().copied().max().unwrap_or(-1) + 1;
    let in_group: Vec<bool> = (0..k).map(|u| groups.iter().any(|g| g.contains(&u))).collect();
    // Untouched units first (in order), then the merged ones
    let keep: Vec<usize> = (0..k).filter(|&u| !in_group[u]).collect();
    let mut new_units: Vec<Vec<usize>> = keep.iter().map(|&u| units[u].clone()).collect();
    let mut data: Vec<f32> = keep.iter().flat_map(|&u| t.data[u * w * m..(u + 1) * w * m].iter().copied()).collect();
    let mut sparsity: Vec<bool> = keep.iter().flat_map(|&u| t.sparsity[u * m..(u + 1) * m].iter().copied()).collect();
    let mut ids: Vec<i64> = keep.iter().map(|&u| t.unit_ids[u]).collect();
    for (gi, g) in groups.iter().enumerate() {
        let mut spikes: Vec<usize> = g.iter().flat_map(|&u| units[u].iter().copied()).collect();
        spikes.sort_by_key(|&i| (samples[i], i));
        let mut kept = Vec::with_capacity(spikes.len());
        for (n, &i) in spikes.iter().enumerate() {
            if n == 0 || samples[i] - samples[spikes[n - 1]] >= rpv {
                kept.push(i);
            }
        }
        let counts: Vec<f64> = g.iter().map(|&u| units[u].len() as f64).collect();
        let total: f64 = counts.iter().sum();
        let mask: Vec<bool> = (0..m).map(|c| g.iter().all(|&u| t.sparsity[u * m + c])).collect();
        for s in 0..w {
            for c in 0..m {
                let v: f64 = if mask[c] { g.iter().zip(&counts).map(|(&u, n)| t.data[(u * w + s) * m + c] as f64 * n / total).sum() } else { 0.0 };
                data.push(v as f32);
            }
        }
        sparsity.extend_from_slice(&mask);
        ids.push(next_id + gi as i64);
        new_units.push(kept);
    }
    let old_index: Vec<Option<usize>> = (0..keep.len()).map(|i| Some(keep[i])).chain(groups.iter().map(|_| None)).collect();
    let n = ids.len();
    let nt = Templates { unit_ids: ids, width: w, channels: m, data, sparsity };
    // Similarities: kept pairs from before, merged rows anew against all
    let mut new_sim = vec![0.0f64; n * n];
    for a in 0..n {
        for b in 0..n {
            if let (Some(oa), Some(ob)) = (old_index[a], old_index[b]) {
                new_sim[a * n + b] = sim[oa * k + ob];
            }
        }
    }
    for a in keep.len()..n {
        for b in 0..n {
            let s = similarity_full(&nt, a, b, opts.num_shifts);
            new_sim[a * n + b] = s;
            new_sim[b * n + a] = s;
        }
    }
    let mut new_loc: Vec<[f64; 2]> = keep.iter().map(|&u| loc[u]).collect();
    for a in keep.len()..n {
        new_loc.push(centre_of_mass(&nt, a, positions));
    }
    *units = new_units;
    *t = nt;
    *sim = new_sim;
    *loc = new_loc;
}

/// Similarity of units `a` (source) and `b` with every lag computed (an array compared with another:
/// no symmetric fill), the best of `±num_shifts`.
fn similarity_full(t: &Templates, a: usize, b: usize, num_shifts: usize) -> f64 {
    let (w, m) = (t.width, t.channels);
    let (sa, sb) = (&t.sparsity[a * m..(a + 1) * m], &t.sparsity[b * m..(b + 1) * m]);
    if !(0..m).any(|c| sa[c] && sb[c]) {
        return 0.0;
    }
    let chans: Vec<usize> = (0..m).filter(|&c| sa[c] || sb[c]).collect();
    let span = w - 2 * num_shifts;
    let (ta, tb) = (&t.data[a * w * m..(a + 1) * w * m], &t.data[b * w * m..(b + 1) * w * m]);
    let mut best = f64::INFINITY;
    for shift in -(num_shifts as i64)..=num_shifts as i64 {
        let (mut diff, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for s in 0..span {
            let (ia, ib) = (s + num_shifts, (s as i64 + num_shifts as i64 + shift) as usize);
            for &c in &chans {
                let (x, y) = (ta[ia * m + c] as f64, tb[ib * m + c] as f64);
                diff += (x - y).abs();
                na += x.abs();
                nb += y.abs();
            }
        }
        best = best.min(diff / (na + nb));
    }
    1.0 - best
}

/// Peak-to-peak centre of mass of unit `u`'s template over its channels.
pub fn centre_of_mass(t: &Templates, u: usize, positions: &[[f32; 2]]) -> [f64; 2] {
    let (w, m) = (t.width, t.channels);
    let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
    for c in (0..m).filter(|&c| t.sparsity[u * m + c]) {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for s in 0..w {
            let v = t.data[(u * w + s) * m + c];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let ptp = (hi - lo) as f64;
        sx += ptp * positions[c][0] as f64;
        sy += ptp * positions[c][1] as f64;
        sw += ptp;
    }
    [sx / sw, sy / sw]
}

/// `get_non_duplicated_events` of two trains: the union sorted (stable), keeping an event when it is
/// more than `delta` after the previous one or from the same train as it.
fn non_duplicated(a: &[i64], b: &[i64], delta: i64) -> Vec<i64> {
    let mut all: Vec<(i64, u8)> = a.iter().map(|&s| (s, 0)).chain(b.iter().map(|&s| (s, 1))).collect();
    all.sort_by_key(|e| e.0);
    let mut out = Vec::with_capacity(all.len());
    for (n, e) in all.iter().enumerate() {
        if n == 0 || e.0 - all[n - 1].0 > delta || e.1 == all[n - 1].1 {
            out.push(e.0);
        }
    }
    out
}

/// Refractory contamination of a sorted train (module docs); violations: pairs within `t_r`.
fn rp_contamination(train: &[i64], total: f64, t_c: f64, t_r: f64) -> f64 {
    let n = train.len();
    if n <= 1 {
        return f64::NAN;
    }
    let mut nv = 0u64;
    for i in 0..n {
        for j in i + 1..n {
            if (train[j] - train[i]) as f64 > t_r {
                break;
            }
            nv += 1;
        }
    }
    let nf = n as f64;
    let denom = 1.0 - nv as f64 * (total - 2.0 * nf * t_c) / (nf * nf * (t_r - t_c));
    if denom < 0.0 { 1.0 } else { 1.0 - denom.sqrt() }
}

/// Upstream `_get_border_probabilities`.
fn borders(max_time: f64) -> (i64, i64, f64, f64) {
    let high = max_time.ceil();
    let low = max_time.floor();
    let p_high = 0.5 * (max_time - high + 1.0).powi(2);
    let mut p_low = 0.5 * (1.0 - (max_time - low).powi(2)) + (max_time - low);
    if low == 0.0 {
        p_low -= 0.5 * (-max_time + 1.0).powi(2);
    }
    (low as i64, high as i64, p_low, p_high)
}

/// Upstream `compute_nb_coincidence`.
fn coincidences(a: &[i64], b: &[i64], max_time: f64) -> f64 {
    if max_time <= 0.0 {
        return 0.0;
    }
    let (low, high, p_low, p_high) = borders(max_time);
    let (mut n, mut n_low, mut n_high) = (0u64, 0u64, 0u64);
    let mut start = 0usize;
    for &x in a {
        let mut j = start;
        while j < b.len() {
            let diff = x - b[j];
            if diff > high {
                start += 1;
                j += 1;
                continue;
            }
            if diff < -high {
                break;
            }
            if diff.abs() == high {
                n_high += 1;
            } else if diff.abs() == low {
                n_low += 1;
            } else {
                n += 1;
            }
            j += 1;
        }
    }
    n as f64 + p_high * n_high as f64 + p_low * n_low as f64
}

/// Upstream `estimate_cross_contamination`'s p-value (`limit = cc_thresh`).
fn cross_contamination_p(a: &[i64], b: &[i64], fs: f64, total: f64, opts: &AutoMergeOptions, c1: f64) -> f64 {
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let t_c = (opts.censored_period_ms * 1e-3 * fs).round();
    let t_r = (opts.refractory_period_ms * 1e-3 * fs).round();
    let violations = coincidences(a, b, t_r) - coincidences(a, b, t_c);
    let n = n1 * n2 * ((1.0 - c1) * opts.cc_thresh + c1);
    let p = 2.0 * t_r / total;
    binom_sf_interp((violations - 1.0) as i64, n, p)
}

/// Upstream `binom_sf`: `binom.sf(x, ·, p)` at the integers `⌊n − 2⌋ .. ⌈n + 3⌉` (non-negative),
/// interpolated at `n` by SciPy's quadratic interpolating spline.
pub fn binom_sf_interp(x: i64, n: f64, p: f64) -> f64 {
    let xs: Vec<f64> = ((n - 2.0).floor() as i64..(n + 3.0).ceil() as i64).filter(|&v| v >= 0).map(|v| v as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&ni| binom_sf(x, ni, p)).collect();
    quadratic_spline(&xs, &ys, n)
}

/// `scipy.stats.binom.sf(x, n, p)` = `P(X > x)` for integer `n`.
fn binom_sf(x: i64, n: f64, p: f64) -> f64 {
    if x < 0 {
        return 1.0;
    }
    if x as f64 >= n {
        return 0.0;
    }
    // P(X > x) = I_p(x + 1, n − x)
    betainc(x as f64 + 1.0, n - x as f64, p)
}

/// Regularised incomplete beta `I_x(a, b)` (continued fraction, Numerical Recipes `betai`).
fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    if x < (a + 1.0) / (a + b + 2.0) {
        (ln_front.exp() * betacf(a, b, x) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - ln_front.exp() * betacf(b, a, 1.0 - x) / b).clamp(0.0, 1.0)
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..200_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// `ln Γ(x)` (Lanczos, g = 7, n = 9), `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// SciPy `interp1d(xs, ys, kind="quadratic")(at)`: the degree-2 interpolating B-spline with
/// `make_interp_spline`'s knots for k = 2 (end knots tripled, the midpoints between the inner points
/// in between).
fn quadratic_spline(xs: &[f64], ys: &[f64], at: f64) -> f64 {
    let n = xs.len();
    if n < 3 {
        // Too few points for a quadratic (n tiny): linear
        if n == 1 {
            return ys[0];
        }
        let f = (at - xs[0]) / (xs[1] - xs[0]);
        return ys[0] + f * (ys[1] - ys[0]);
    }
    let mids: Vec<f64> = xs.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    let mut knots = vec![xs[0]; 3];
    knots.extend_from_slice(&mids[1..mids.len() - 1]);
    knots.extend([xs[n - 1]; 3]);
    // Collocation matrix: B_j(x_i)
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = bspline_basis(&knots, j, 2, xs[i], i == n - 1);
        }
    }
    let coef = solve(&mut a, ys.to_vec(), n);
    (0..n).map(|j| coef[j] * bspline_basis(&knots, j, 2, at, at >= xs[n - 1])).sum()
}

/// Cox–de Boor basis `B_{j,k}(x)`; `last`: the right end belongs to the last interval.
fn bspline_basis(t: &[f64], j: usize, k: usize, x: f64, last: bool) -> f64 {
    if k == 0 {
        let inside = (t[j] <= x && x < t[j + 1]) || (last && x == t[j + 1] && t[j] < t[j + 1] && t[j + 1] == t[t.len() - 1]);
        return if inside { 1.0 } else { 0.0 };
    }
    let mut v = 0.0;
    let d1 = t[j + k] - t[j];
    if d1 > 0.0 {
        v += (x - t[j]) / d1 * bspline_basis(t, j, k - 1, x, last);
    }
    let d2 = t[j + k + 1] - t[j + 1];
    if d2 > 0.0 {
        v += (t[j + k + 1] - x) / d2 * bspline_basis(t, j + 1, k - 1, x, last);
    }
    v
}

/// Gaussian elimination with partial pivoting (small systems).
fn solve(a: &mut [f64], mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        let piv = (col..n).max_by(|&r, &s| a[r * n + col].abs().total_cmp(&a[s * n + col].abs())).unwrap_or(col);
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] / d;
            if f != 0.0 {
                for c in col..n {
                    a[r * n + c] -= f * a[col * n + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    for r in (0..n).rev() {
        let mut v = b[r];
        for c in r + 1..n {
            v -= a[r * n + c] * b[c];
        }
        b[r] = v / a[r * n + r];
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values from SciPy 1.16 (`binom.sf`, and upstream's `binom_sf` interpolation).
    #[test]
    fn binomial_tail_matches_scipy() {
        assert!((binom_sf(3, 10.0, 0.2) - 0.120_873_881_6).abs() < 1e-9, "{}", binom_sf(3, 10.0, 0.2));
        assert!((binom_sf(0, 5.0, 0.5) - 0.968_75).abs() < 1e-12);
        assert_eq!(binom_sf(-1, 5.0, 0.5), 1.0);
        // A large n, small p: Poisson-like; sf(9; 1e6, 1e-5) ≈ P(Poisson(10) > 9) = 0.54207
        assert!((binom_sf(9, 1e6, 1e-5) - 0.542_070_911_061_030_4).abs() < 1e-9, "{}", binom_sf(9, 1e6, 1e-5));
        // Upstream's interpolated `binom_sf` at non-integer n (SciPy `interp1d(kind="quadratic")`)
        for (x, n, p, want) in [(5, 1234.6, 0.004, 0.373_210_018_512_101_2), (40, 98765.3, 3e-4, 0.027_436_597_478_162_36), (0, 2.5, 0.3, 0.590_050_312_5)] {
            let got = binom_sf_interp(x, n, p);
            assert!((got - want).abs() < 1e-9 * want.max(1e-3), "binom_sf({x}, {n}, {p}) = {got}, scipy {want}");
        }
        // The quadratic spline reproduces a quadratic exactly
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|x| x * x - 3.0 * x).collect();
        assert!((quadratic_spline(&xs, &ys, 2.5) - (6.25 - 7.5)).abs() < 1e-12);
        assert!((quadratic_spline(&xs, &ys, 5.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn contamination_and_coincidences() {
        // A clean train: 1000 spikes every 1000 samples, no violations
        let clean: Vec<i64> = (0..1000).map(|i| i * 1000).collect();
        assert_eq!(rp_contamination(&clean, 1e6, 9.0, 30.0), 0.0);
        // Coincidences within 30 samples between two identical trains: every pair (diff 0)
        assert_eq!(coincidences(&clean, &clean, 30.0), 1000.0);
        let dup = non_duplicated(&clean, &clean.iter().map(|s| s + 5).collect::<Vec<_>>(), 9);
        assert_eq!(dup.len(), 1000, "the shifted copy is within the censored period");
    }

    /// Two halves of one unit merge; a distinct unit does not.
    #[test]
    fn halves_of_a_unit_merge() {
        let (w, m) = (30usize, 2usize);
        let wave: Vec<f32> = (0..w).map(|s| -10.0 * (-((s as f32 - 10.0) / 2.0).powi(2)).exp()).collect();
        let mut data = vec![0.0f32; 3 * w * m];
        for s in 0..w {
            data[s * m] = wave[s];
            data[w * m + s * m] = wave[s] * 1.02;
            data[2 * w * m + s * m + 1] = wave[s];
        }
        let sparsity = vec![true, false, true, false, false, true];
        let t = Templates { unit_ids: vec![0, 1, 2], width: w, channels: m, data, sparsity };
        let fs = 30_000.0;
        let total = 3_000_000u64;
        // One neuron firing every 2000 samples, its spikes split alternately; unit 2 independent
        let mut samples = Vec::new();
        let mut units = vec![Vec::new(), Vec::new(), Vec::new()];
        for i in 0..1400u64 {
            units[(i % 2) as usize].push(samples.len());
            samples.push(i * 2000 + 37);
        }
        for i in 0..1000u64 {
            units[2].push(samples.len());
            samples.push(i * 2900 + 11);
        }
        let mut order: Vec<usize> = (0..samples.len()).collect();
        order.sort_by_key(|&i| samples[i]);
        for u in units.iter_mut() {
            u.sort_by_key(|&i| samples[i]);
        }
        let pos = vec![[0.0f32, 0.0], [0.0, 30.0]];
        let out = auto_merge(&samples, units, t, &pos, fs, total, &AutoMergeOptions::default());
        assert_eq!(out.templates.len(), 2, "{:?}", out.templates.unit_ids);
        assert_eq!(out.templates.unit_ids, vec![2, 3]);
        assert_eq!(out.spikes[1].len(), 1400);
    }
}
