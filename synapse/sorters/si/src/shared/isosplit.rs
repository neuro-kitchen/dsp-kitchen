//! SpikeInterface's isosplit (`sortingcomponents/clustering/isosplit_isocut.py`, MIT, Copyright
//! (c) 2019 SpikeInterface): its Python port of `magland/isosplit6` (Magland & Barnett, ISO-SPLIT,
//! arXiv:1508.04841), quirks included, used by Tridesclous 2's splits. MountainSort 5 uses the C++
//! original, a separate copy in `dspk-sorter-ms5`.
//!
//! 1. **Initial labels**: SciPy's `kmeans2(minit="points")` ([`kmeans2_points`], [`isosplit_si`]),
//!    or labels given ([`isosplit_from_labels`]).
//! 2. **Passes**: every cluster is paired with its mutual nearest centroid among the pairs not
//!    compared in this pass. A pair merges when either is smaller than `min_cluster_size`, or when
//!    [`isocut`] finds no dip (score below `isocut_threshold`) in the points' projection on
//!    `(Σ₁ + Σ₂)⁻¹ (μ₂ − μ₁)` (covariances averaged); otherwise the points are redistributed at
//!    the cut point. Changed clusters get new centroids and covariances. Passes repeat until one
//!    merges nothing.
//!
//! Where SpikeInterface differs from the C++ original (kept, to match its results): each
//! covariance's diagonal is counted twice; the loop starts on the final pass (a first pass that
//! merges nothing ends it); the side below the cut is labelled 2; a redistribution whose moves add
//! up to a whole cluster is skipped ("pure swapping"); the up-down isotonic fit gives the split
//! index to the right part; a dip-score tie between the two sides picks the right one.
//!
//! [`isocut`]: the 1-D dip test. Sorted values; log-densities between neighbours `log(1 /
//! spacing)`; the best unimodal fit by up-down isotonic regression; the dip score is the largest
//! Kolmogorov–Smirnov distance (scaled by `√(n/2)`) between data and fit over ranges halving from
//! the peak on either side; the cut point is where the residual's down-up fit is lowest.
//!
//! On the host, in `f64` as upstream. Choice of ours: when the averaged covariance cannot be
//! inverted a ridge of `RIDGE` × the mean diagonal is added and the inversion retried.

/// Settings of [`isosplit_from_labels`] (SpikeInterface's `isosplit` keywords).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsosplitOptions {
    /// Dip scores below this merge a pair.
    pub isocut_threshold: f64,
    /// Smaller clusters always merge.
    pub min_cluster_size: usize,
    pub max_iterations_per_pass: usize,
}

impl Default for IsosplitOptions {
    fn default() -> Self {
        Self { isocut_threshold: 2.0, min_cluster_size: 10, max_iterations_per_pass: 500 }
    }
}

/// Log-density of a zero spacing (upstream `log(0.000000001)`).
const ZERO_SPACING_DENSITY: f64 = 1e-9;
/// Ridge (× the mean diagonal) added when the averaged covariance is singular.
const RIDGE: f64 = 1e-10;

/// SpikeInterface's `isosplit` (module docs) from
/// `initial` labels (any values; made continuous in increasing order): labels `0..K`.
///
/// # Panics
///
/// If `x.len() != n · m` or `initial.len() != n`.
pub fn isosplit_from_labels(x: &[f64], n: usize, m: usize, initial: &[u32], opts: &IsosplitOptions) -> Vec<u32> {
    assert_eq!(x.len(), n * m, "x must be [n, m]");
    assert_eq!(initial.len(), n, "one initial label per point");
    if n == 0 {
        return Vec::new();
    }
    let mut set: Vec<u32> = initial.to_vec();
    set.sort_unstable();
    set.dedup();
    let labels: Vec<u32> = initial.iter().map(|l| set.binary_search(l).unwrap() as u32 + 1).collect();
    iterate(x, m, labels, opts).into_iter().map(|l| l - 1).collect()
}

/// SciPy `kmeans2(x, k, minit="points", seed)` (10 iterations): `k` distinct points drawn at
/// random start the centroids; each iteration assigns every point to its nearest centroid (the
/// first on ties) and moves each centroid to its points' mean (an empty one stays). Returns the
/// last assignment. Our random stream, not NumPy's.
pub fn kmeans2_points(x: &[f64], n: usize, m: usize, k: usize, seed: u64) -> Vec<u32> {
    let k = k.clamp(1, n.max(1));
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (next() % (n - i) as u64) as usize;
        pool.swap(i, j);
    }
    let mut centroids: Vec<f64> = pool[..k].iter().flat_map(|&i| x[i * m..(i + 1) * m].iter().copied()).collect();
    let mut labels = vec![0u32; n];
    for _ in 0..10 {
        for i in 0..n {
            let p = &x[i * m..(i + 1) * m];
            let mut best = (0u32, f64::INFINITY);
            for c in 0..k {
                let d: f64 = (0..m).map(|j| (p[j] - centroids[c * m + j]).powi(2)).sum();
                if d < best.1 {
                    best = (c as u32, d);
                }
            }
            labels[i] = best.0;
        }
        let mut sums = vec![0.0f64; k * m];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i] as usize;
            counts[c] += 1;
            for j in 0..m {
                sums[c * m + j] += x[i * m + j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..m {
                    centroids[c * m + j] = sums[c * m + j] / counts[c] as f64;
                }
            }
        }
    }
    labels
}

/// SpikeInterface's `isosplit(X, n_init, ...)`: `n_init` lowered when too large for the sample
/// (`max(1, n / (2 · min_cluster_size))`), [`kmeans2_points`], then [`isosplit_from_labels`].
pub fn isosplit_si(x: &[f64], n: usize, m: usize, n_init: usize, seed: u64, opts: &IsosplitOptions) -> Vec<u32> {
    let mcs = opts.min_cluster_size.max(1);
    let n_init = if n_init >= n || n_init > n / mcs { (n / (2 * mcs)).max(1) } else { n_init };
    let init = kmeans2_points(x, n, m, n_init, seed);
    isosplit_from_labels(x, n, m, &init, opts)
}

/// The passes and iterations from labels `1..=K` (module docs); labels `1..=K'` renumbered.
fn iterate(x: &[f64], m: usize, mut labels: Vec<u32>, opts: &IsosplitOptions) -> Vec<u32> {
    let kmax = *labels.iter().max().unwrap_or(&0) as usize;
    let all = vec![true; kmax];
    let mut centroids = vec![0.0; m * kmax];
    let mut covmats = vec![0.0; m * m * kmax];
    let mut members = members_of(&labels, kmax);
    compute_centroids(&mut centroids, x, m, &members, &all);
    compute_covmats(&mut covmats, x, m, &members, &centroids, &all);
    let mut active: Vec<usize> = (1..=kmax).filter(|&k| !members[k - 1].is_empty()).collect();
    let mut compared = vec![false; kmax * kmax];
    // SpikeInterface starts on the final pass: a first pass that merges nothing ends the loop
    let mut final_pass = true;
    loop {
        let mut something_merged = false;
        let mut changed_in_pass = vec![false; kmax];
        let mut iteration = 0;
        loop {
            iteration += 1;
            if iteration > opts.max_iterations_per_pass {
                break;
            }
            if active.is_empty() {
                break;
            }
            let pairs = pairs_to_compare(&centroids, m, &active, &compared, kmax);
            if pairs.is_empty() {
                break;
            }
            let changed = compare_pairs(x, m, &mut labels, &members, &pairs, opts, &centroids, &covmats, kmax);
            members = members_of(&labels, kmax);
            let mut changed_in_iteration = vec![false; kmax];
            for &k in &changed {
                changed_in_pass[k - 1] = true;
                changed_in_iteration[k - 1] = true;
            }
            for &(k1, k2) in &pairs {
                compared[(k1 - 1) * kmax + (k2 - 1)] = true;
                compared[(k2 - 1) * kmax + (k1 - 1)] = true;
            }
            compute_centroids(&mut centroids, x, m, &members, &changed_in_iteration);
            compute_covmats(&mut covmats, x, m, &members, &centroids, &changed_in_iteration);
            let new_active: Vec<usize> = (1..=kmax).filter(|&k| !members[k - 1].is_empty()).collect();
            if new_active.len() < active.len() {
                something_merged = true;
            }
            active = new_active;
        }
        for k in 0..kmax {
            if changed_in_pass[k] {
                for j in 0..kmax {
                    compared[k * kmax + j] = false;
                    compared[j * kmax + k] = false;
                }
            }
        }
        if something_merged {
            final_pass = false;
        }
        if final_pass {
            break;
        }
        if !something_merged {
            final_pass = true;
        }
    }
    let mut map = vec![0u32; kmax];
    for (i, &k) in active.iter().enumerate() {
        map[k - 1] = i as u32 + 1;
    }
    labels.iter().map(|&l| map[l as usize - 1]).collect()
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum::<f64>().sqrt()
}

/// The points of each label (`members[k − 1]`), in index order.
fn members_of(labels: &[u32], kmax: usize) -> Vec<Vec<usize>> {
    let mut members = vec![Vec::new(); kmax];
    for (i, &l) in labels.iter().enumerate() {
        members[l as usize - 1].push(i);
    }
    members
}

/// Centroids of the clusters flagged in `which` (label `k` → row `k − 1`); an empty cluster gets 0.
fn compute_centroids(centroids: &mut [f64], x: &[f64], m: usize, members: &[Vec<usize>], which: &[bool]) {
    for (k, idx) in members.iter().enumerate() {
        if !which[k] {
            continue;
        }
        let c = &mut centroids[k * m..(k + 1) * m];
        c.iter_mut().for_each(|v| *v = 0.0);
        for &i in idx {
            for d in 0..m {
                c[d] += x[i * m + d];
            }
        }
        if !idx.is_empty() {
            c.iter_mut().for_each(|v| *v /= idx.len() as f64);
        }
    }
}

/// Points per partial sum of a covariance (the partial sums run in parallel).
const COV_CHUNK: usize = 2048;

/// Covariances (divided by the count) of the clusters flagged in `which`.
fn compute_covmats(covmats: &mut [f64], x: &[f64], m: usize, members: &[Vec<usize>], centroids: &[f64], which: &[bool]) {
    use rayon::prelude::*;
    for (k, idx) in members.iter().enumerate() {
        if !which[k] {
            continue;
        }
        let c = &centroids[k * m..(k + 1) * m];
        // Upper triangle, in chunks summed in parallel
        let upper = idx
            .par_chunks(COV_CHUNK)
            .map(|chunk| {
                let mut acc = vec![0.0; m * m];
                let mut diff = vec![0.0; m];
                for &i in chunk {
                    for d in 0..m {
                        diff[d] = x[i * m + d] - c[d];
                    }
                    for a in 0..m {
                        for b in a..m {
                            acc[a * m + b] += diff[a] * diff[b];
                        }
                    }
                }
                acc
            })
            // Partials collected in chunk order and added in that order: the same sums every run
            .collect::<Vec<_>>()
            .into_iter()
            .fold(vec![0.0; m * m], |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(u, v)| *u += v);
                a
            });
        let n = idx.len().max(1) as f64;
        let cov = &mut covmats[k * m * m..(k + 1) * m * m];
        for a in 0..m {
            for b in a..m {
                let v = if idx.is_empty() { 0.0 } else { upper[a * m + b] / n };
                cov[a * m + b] = v;
                cov[b * m + a] = v;
            }
            // SpikeInterface adds each diagonal term twice (its loop over m2 ≥ m1 writes both halves)
            cov[a * m + a] *= 2.0;
        }
    }
}

/// Upstream `get_pairs_to_compare`: mutual nearest active centroids among pairs not compared yet.
fn pairs_to_compare(centroids: &[f64], m: usize, active: &[usize], compared: &[bool], kmax: usize) -> Vec<(usize, usize)> {
    let k = active.len();
    let mut dists = vec![-1.0; k * k];
    for a in 0..k {
        for b in 0..k {
            if a != b && !compared[(active[a] - 1) * kmax + (active[b] - 1)] {
                dists[a * k + b] = dist(&centroids[(active[a] - 1) * m..active[a] * m], &centroids[(active[b] - 1) * m..active[b] * m]);
            }
        }
    }
    let best: Vec<Option<usize>> = (0..k)
        .map(|a| {
            let mut best = None;
            let mut best_d = -1.0;
            for b in 0..k {
                let d = dists[a * k + b];
                if d >= 0.0 && (best_d < 0.0 || d < best_d) {
                    best_d = d;
                    best = Some(b);
                }
            }
            best
        })
        .collect();
    let mut pairs = Vec::new();
    for j in 0..k {
        if let Some(b) = best[j] {
            if b > j && best[b] == Some(j) && dists[j * k + b] >= 0.0 {
                pairs.push((active[j], active[b]));
                for aa in 0..k {
                    dists[j * k + aa] = -1.0;
                    dists[aa * k + j] = -1.0;
                    dists[b * k + aa] = -1.0;
                    dists[aa * k + b] = -1.0;
                }
            }
        }
    }
    pairs
}

/// Upstream `compare_pairs`: merges or redistributes each pair; returns the changed labels.
#[allow(clippy::too_many_arguments)]
fn compare_pairs(
    x: &[f64],
    m: usize,
    labels: &mut [u32],
    members: &[Vec<usize>],
    pairs: &[(usize, usize)],
    opts: &IsosplitOptions,
    centroids: &[f64],
    covmats: &[f64],
    kmax: usize,
) -> Vec<usize> {
    use rayon::prelude::*;
    let mut changed = vec![false; kmax];
    let mut new_labels = labels.to_vec();
    // Each pair's test, in parallel (the pairs are disjoint); applied in order below
    let tests: Vec<Option<(bool, Vec<u8>)>> = pairs
        .par_iter()
        .map(|&(k1, k2)| {
            let (inds1, inds2) = (&members[k1 - 1], &members[k2 - 1]);
            if inds1.is_empty() || inds2.is_empty() {
                return None;
            }
            if inds1.len() < opts.min_cluster_size || inds2.len() < opts.min_cluster_size {
                return Some((true, Vec::new()));
            }
            let c1 = &centroids[(k1 - 1) * m..k1 * m];
            let c2 = &centroids[(k2 - 1) * m..k2 * m];
            let avg: Vec<f64> = (0..m * m).map(|e| (covmats[(k1 - 1) * m * m + e] + covmats[(k2 - 1) * m * m + e]) / 2.0).collect();
            let inv = invert(&avg, m);
            let diff: Vec<f64> = (0..m).map(|d| c2[d] - c1[d]).collect();
            let mut v: Vec<f64> = (0..m).map(|a| (0..m).map(|b| inv[a * m + b] * diff[b]).sum()).collect();
            let norm = v.iter().map(|t| t * t).sum::<f64>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|t| *t /= norm);
            }
            let proj: Vec<f64> = inds1.iter().chain(inds2.iter()).map(|&i| (0..m).map(|d| v[d] * x[i * m + d]).sum()).collect();
            let (dipscore, cutpoint) = isocut(&proj);
            // SpikeInterface labels the side below the cut 2 (`(proj < cut) + 1`)
            let side = proj.iter().map(|&p| if p < cutpoint { 2u8 } else { 1 }).collect();
            Some((dipscore < opts.isocut_threshold, side))
        })
        .collect();
    for (&(k1, k2), test) in pairs.iter().zip(tests) {
        let Some((merge, side)) = test else { continue };
        let (inds1, inds2) = (&members[k1 - 1], &members[k2 - 1]);
        if merge {
            for &i in inds2.iter() {
                new_labels[i] = k1 as u32;
            }
            changed[k1 - 1] = true;
            changed[k2 - 1] = true;
        } else {
            // "Pure swapping" guard: no redistribution when the moves add up to a whole cluster
            let m1 = (0..inds1.len()).filter(|&j| side[j] == 2).count();
            let m2 = (0..inds2.len()).filter(|&j| side[inds1.len() + j] == 1).count();
            if m1 as f64 / inds1.len() as f64 + m2 as f64 / inds2.len() as f64 >= 1.0 {
                continue;
            }
            let mut moved = false;
            for (j, &i) in inds1.iter().enumerate() {
                if side[j] == 2 {
                    new_labels[i] = k2 as u32;
                    moved = true;
                }
            }
            for (j, &i) in inds2.iter().enumerate() {
                if side[inds1.len() + j] == 1 {
                    new_labels[i] = k1 as u32;
                    moved = true;
                }
            }
            if moved {
                changed[k1 - 1] = true;
                changed[k2 - 1] = true;
            }
        }
    }
    labels.copy_from_slice(&new_labels);
    (1..=kmax).filter(|&k| changed[k - 1]).collect()
}

/// Inverse of the symmetric `[m, m]` matrix `a` (Gauss–Jordan, partial pivoting); a ridge is added
/// and the inversion retried when it is singular (module docs).
fn invert(a: &[f64], m: usize) -> Vec<f64> {
    let mean_diag = (0..m).map(|i| a[i * m + i].abs()).sum::<f64>() / m.max(1) as f64;
    let mut ridge = 0.0;
    loop {
        let mut aug = a.to_vec();
        for i in 0..m {
            aug[i * m + i] += ridge;
        }
        if let Some(inv) = gauss_jordan(&mut aug, m) {
            return inv;
        }
        ridge = if ridge == 0.0 { RIDGE * mean_diag.max(f64::MIN_POSITIVE) } else { ridge * 10.0 };
    }
}

fn gauss_jordan(a: &mut [f64], m: usize) -> Option<Vec<f64>> {
    let mut inv: Vec<f64> = (0..m * m).map(|e| if e / m == e % m { 1.0 } else { 0.0 }).collect();
    for col in 0..m {
        let piv = (col..m).max_by(|&r, &s| a[r * m + col].abs().total_cmp(&a[s * m + col].abs()))?;
        if a[piv * m + col].abs() < f64::EPSILON * 1e-3 || !a[piv * m + col].is_finite() {
            return None;
        }
        for c in 0..m {
            a.swap(col * m + c, piv * m + c);
            inv.swap(col * m + c, piv * m + c);
        }
        let d = a[col * m + col];
        for c in 0..m {
            a[col * m + c] /= d;
            inv[col * m + c] /= d;
        }
        for r in 0..m {
            if r != col {
                let f = a[r * m + col];
                if f != 0.0 {
                    for c in 0..m {
                        a[r * m + c] -= f * a[col * m + c];
                        inv[r * m + c] -= f * inv[col * m + c];
                    }
                }
            }
        }
    }
    Some(inv)
}

/// `(dip score, cut point)` of 1-D samples (SpikeInterface's `isocut`, module docs).
pub fn isocut(samples: &[f64]) -> (f64, f64) {
    let n = samples.len();
    let mut x = samples.to_vec();
    x.sort_by(f64::total_cmp);
    if n < 2 {
        return (0.0, x.first().copied().unwrap_or(0.0));
    }
    let spacings: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let multiplicities = vec![1.0; n - 1];
    let log_densities: Vec<f64> = spacings.iter().map(|&s| if s != 0.0 { (1.0 / s).ln() } else { ZERO_SPACING_DENSITY.ln() }).collect();
    let fit = isotonic_updown(&log_densities, &multiplicities);
    let fit_times_spacings: Vec<f64> = fit.iter().zip(&spacings).map(|(f, s)| f.exp() * s).collect();
    let peak = index_of_max(&fit);
    let (dipscore, lo, hi) = ks5(&multiplicities, &fit_times_spacings, peak);
    let resid: Vec<f64> = (lo..=hi).map(|i| log_densities[i] - fit[i]).collect();
    let resid_fit = isotonic_downup(&resid, &vec![1.0; resid.len()]);
    let cut = index_of_min(&resid_fit);
    (dipscore, (x[lo + cut] + x[lo + cut + 1]) / 2.0)
}

/// First index of the largest value.
fn index_of_max(v: &[f64]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// First index of the smallest value.
fn index_of_min(v: &[f64]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] < v[b] { i } else { b })
}

/// Upstream `compute_ks4`: largest difference of the normalised cumulative sums, scaled.
fn ks4(c1: &[f64], c2: &[f64]) -> f64 {
    let (s1, s2): (f64, f64) = (c1.iter().sum(), c2.iter().sum());
    let (mut cum1, mut cum2, mut best) = (0.0, 0.0, 0.0f64);
    for i in 0..c1.len() {
        cum1 += c1[i];
        cum2 += c2[i];
        if s1 > 0.0 && s2 > 0.0 {
            best = best.max((cum1 / s1 - cum2 / s2).abs());
        }
    }
    best * ((s1 + s2) / 2.0).sqrt()
}

/// Upstream `compute_ks5`: `(score, critical range lo, hi)` over ranges halving from the peak.
fn ks5(c1: &[f64], c2: &[f64], peak: usize) -> (f64, usize, usize) {
    let n = c1.len();
    let (mut lo, mut hi, mut best) = (0, n - 1, -1.0);
    let mut left_best = f64::NEG_INFINITY;
    let mut len = peak + 1;
    loop {
        let score = ks4(&c1[..len], &c2[..len]);
        if score > best {
            (lo, hi, best) = (0, len - 1, score);
        }
        len /= 2;
        if !(len >= 4 || len == peak + 1) {
            break;
        }
    }
    left_best = left_best.max(best);
    let mut right_best = f64::NEG_INFINITY;
    let (mut right_lo, mut right_hi) = (0, n - 1);
    let r1: Vec<f64> = c1.iter().rev().copied().collect();
    let r2: Vec<f64> = c2.iter().rev().copied().collect();
    let mut len = n - peak;
    loop {
        let score = ks4(&r1[..len], &r2[..len]);
        if score > right_best {
            (right_lo, right_hi, right_best) = (n - len, n - 1, score);
        }
        len /= 2;
        if !(len >= 4 || len == n - peak) {
            break;
        }
    }
    // The right side wins unless the left is strictly better
    if !(left_best > right_best) {
        (right_best, right_lo, right_hi)
    } else {
        (left_best, lo, hi)
    }
}

/// Upstream `jisotonic5`: the non-decreasing least-squares fit of `a` (weights `w`) by pooling
/// adjacent violators, and the fit's squared error over every prefix.
fn isotonic(a: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = a.len();
    let mut out = vec![0.0; n];
    let mut mse = vec![0.0; n];
    if n == 0 {
        return (out, mse);
    }
    let (mut count, mut sum, mut sumsqr, mut members): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) =
        (vec![w[0]], vec![a[0] * w[0]], vec![a[0] * a[0] * w[0]], vec![1]);
    for j in 1..n {
        count.push(w[j]);
        sum.push(a[j] * w[j]);
        sumsqr.push(a[j] * a[j] * w[j]);
        members.push(1);
        mse[j] = mse[j - 1];
        while count.len() > 1 {
            let last = count.len() - 1;
            if sum[last - 1] / count[last - 1] < sum[last] / count[last] {
                break;
            }
            let prev = sumsqr[last - 1] - sum[last - 1] * sum[last - 1] / count[last - 1] + sumsqr[last] - sum[last] * sum[last] / count[last];
            let (c, s, q, u) = (count.pop().unwrap(), sum.pop().unwrap(), sumsqr.pop().unwrap(), members.pop().unwrap());
            let last = count.len() - 1;
            count[last] += c;
            sum[last] += s;
            sumsqr[last] += q;
            members[last] += u;
            mse[j] += sumsqr[last] - sum[last] * sum[last] / count[last] - prev;
        }
    }
    let mut i = 0;
    for k in 0..count.len() {
        for _ in 0..members[k] {
            out[i] = sum[k] / count[k];
            i += 1;
        }
    }
    (out, mse)
}

/// The best fit rising then falling, SpikeInterface's version: the left fit stops before the best
/// split index, the right one starts at it.
fn isotonic_updown(a: &[f64], w: &[f64]) -> Vec<f64> {
    let n = a.len();
    let ar: Vec<f64> = a.iter().rev().copied().collect();
    let wr: Vec<f64> = w.iter().rev().copied().collect();
    let (_, mse1) = isotonic(a, w);
    let (_, mse2) = isotonic(&ar, &wr);
    let total: Vec<f64> = (0..n).map(|j| mse1[j] + mse2[n - 1 - j]).collect();
    let best = index_of_min(&total);
    let (y1, _) = isotonic(&a[..best], &w[..best]);
    let neg: Vec<f64> = a[best..].iter().map(|v| -v).collect();
    let (y2, _) = isotonic(&neg, &w[best..]);
    y1.into_iter().chain(y2.into_iter().map(|v| -v)).collect()
}

/// The best fit falling then rising.
fn isotonic_downup(a: &[f64], w: &[f64]) -> Vec<f64> {
    let neg: Vec<f64> = a.iter().map(|v| -v).collect();
    isotonic_updown(&neg, w).into_iter().map(|v| -v).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic standard normal draws (Box–Muller on a splitmix64 stream).
    struct Normal(u64);
    impl Normal {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn next(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    /// Against SpikeInterface's `isosplit_isocut.isosplit` run on the same points and initial labels:
    /// set `ISOSPLIT_SI_REF` to the JSON its reference script writes (cases of `x`, `n`, `m`, `init`,
    /// `out`).
    #[test]
    #[ignore = "needs ISOSPLIT_SI_REF (a SpikeInterface run)"]
    fn matches_spikeinterface_upstream() {
        let path = std::env::var("ISOSPLIT_SI_REF").expect("ISOSPLIT_SI_REF");
        let cases: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let opts = IsosplitOptions::default();
        for (ci, c) in cases.iter().enumerate() {
            let x: Vec<f64> = c["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let (n, m) = (c["n"].as_u64().unwrap() as usize, c["m"].as_u64().unwrap() as usize);
            let init: Vec<u32> = c["init"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let want: Vec<u32> = c["out"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let got = isosplit_from_labels(&x, n, m, &init, &opts);
            assert_eq!(got, want, "case {ci}");
        }
    }

    #[test]
    fn isotonic_pools_adjacent_violators() {
        let (fit, mse) = isotonic(&[1.0, 3.0, 2.0, 4.0], &[1.0; 4]);
        assert_eq!(fit, vec![1.0, 2.5, 2.5, 4.0]);
        assert!((mse[3] - 0.5).abs() < 1e-12, "{mse:?}");
        // Up then down: a peak in the middle is kept
        let ud = isotonic_updown(&[1.0, 2.0, 5.0, 3.0, 1.0], &[1.0; 5]);
        assert_eq!(ud, vec![1.0, 2.0, 5.0, 3.0, 1.0]);
    }

    #[test]
    fn isocut_finds_a_dip_between_two_modes_and_none_in_one() {
        let mut rng = Normal(1);
        let one: Vec<f64> = (0..2000).map(|_| rng.next()).collect();
        let (dip, _) = isocut(&one);
        assert!(dip < 2.0, "one mode: dip {dip}");
        let two: Vec<f64> = (0..2000).map(|i| rng.next() + if i % 2 == 0 { -4.0 } else { 4.0 }).collect();
        let (dip, cut) = isocut(&two);
        assert!(dip > 2.0, "two modes: dip {dip}");
        assert!(cut.abs() < 1.5, "cut between the modes: {cut}");
    }

    #[test]
    fn clusters_separated_blobs_and_keeps_one_blob_whole() {
        let mut rng = Normal(7);
        let centres = [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let mut x = Vec::new();
        let mut truth = Vec::new();
        for i in 0..1500 {
            let c = i % 3;
            x.extend([centres[c][0] + rng.next(), centres[c][1] + rng.next()]);
            truth.push(c);
        }
        let opts = IsosplitOptions::default();
        let labels = isosplit_si(&x, 1500, 2, 15, 0, &opts);
        let k = *labels.iter().max().unwrap() + 1;
        assert_eq!(k, 3, "three blobs");
        // Each blob is one cluster
        for c in 0..3 {
            let mut seen: Vec<u32> = (0..1500).filter(|&i| truth[i] == c).map(|i| labels[i]).collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), 1, "blob {c} split: {seen:?}");
        }
        assert_eq!(labels, isosplit_si(&x, 1500, 2, 15, 0, &opts), "deterministic");
        let blob: Vec<f64> = (0..2000).map(|_| rng.next()).collect();
        let one = isosplit_si(&blob, 1000, 2, 15, 0, &opts);
        assert!(one.iter().all(|&l| l == one[0]), "one blob stays one cluster");
    }
}
