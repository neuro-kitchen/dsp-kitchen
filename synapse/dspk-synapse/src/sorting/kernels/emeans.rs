//! e-means kernels. The whole run stays on the device: the epoch's shuffle (a keyed permutation
//! computed per point, nothing uploaded), the batch assignments and sums, the EMA update of the
//! centres and masses, the last epoch's sums and the closing means. Points are feature-major
//! (`[d, n]`, see [`super::points`]); centres row-major (`[k, d]`). A batch is a contiguous range
//! `[start, start + len)` of the shuffled points.

use cubecl::prelude::*;
use dspk_core::compute::element_position;

/// One round function of the shuffle's Feistel network: a 32-bit mix of `x` and the round key.
#[cube]
fn shuffle_mix(x: u32, key: u32) -> u32 {
    let mut h = x ^ key;
    h *= 0x9E37_79B1u32;
    h ^= h >> 15u32;
    h *= 0x85EB_CA6Bu32;
    h ^ (h >> 13u32)
}

/// Position `j < n` of the epoch's shuffle: a 4-round Feistel network on `2 · half` bits (a
/// permutation of `0..2^(2·half) ⊇ 0..n`), walked again while it lands at or past `n` (so a
/// permutation of `0..n`). Must match `permute` on the host bit for bit.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn shuffle_index(j: u32, n: u32, half: u32, mask: u32, k0: u32, k1: u32, k2: u32, k3: u32) -> u32 {
    let mut x = j;
    let mut done = false;
    while !done {
        let mut l = x >> half;
        let mut r = x & mask;
        let t0 = l ^ (shuffle_mix(r, k0) & mask);
        l = r;
        r = t0;
        let t1 = l ^ (shuffle_mix(r, k1) & mask);
        l = r;
        r = t1;
        let t2 = l ^ (shuffle_mix(r, k2) & mask);
        l = r;
        r = t2;
        let t3 = l ^ (shuffle_mix(r, k3) & mask);
        l = r;
        r = t3;
        x = (l << half) | r;
        done = x < n;
    }
    x
}

/// `out[f, j] = x[f, shuffle(j)]` for the first `m` positions of the epoch's shuffle (feature-major
/// `[d, m]`). One unit per position: its feature loop writes neighbouring addresses across units.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gather_shuffled_kernel<F: Float>(x: &[F], out: &mut [F], n: u32, m: u32, d: u32, half: u32, mask: u32, k0: u32, k1: u32, k2: u32, k3: u32) {
    let j = element_position() as u32;
    if j < m {
        let src = shuffle_index(j, n, half, mask, k0, k1, k2, k3);
        let mut f = 0u32;
        while f < d {
            out[(f * m + j) as usize] = x[(f * n + src) as usize];
            f += 1u32;
        }
    }
}

/// Starting centres: `centres[c, f] = x[f, index[c]]` (row-major `[k, d]` from feature-major
/// points). One unit per `(c, f)`.
#[cube(launch)]
pub fn gather_rows_kernel<F: Float>(x: &[F], index: &[u32], centres: &mut [F], n: u32, k: u32, d: u32) {
    let unit = element_position() as u32;
    if unit < k * d {
        let c = unit / d;
        let f = unit - c * d;
        centres[unit as usize] = x[(f * n + index[c as usize]) as usize];
    }
}

/// `label[start + j]` = nearest of the `k` centres (the first on ties) for the `len` points of the
/// batch. One unit per batch point.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn batch_nearest_kernel<F: Float>(x: &[F], centres: &[F], label: &mut [u32], n: u32, start: u32, len: u32, d: u32, k: u32) {
    let j = element_position() as u32;
    if j < len {
        let i = start + j;
        let mut best = F::max_value();
        let mut best_c = 0u32;
        let mut c = 0u32;
        while c < k {
            let mut acc = F::new(0.0f32);
            let mut f = 0u32;
            while f < d {
                let diff = x[(f * n + i) as usize] - centres[(c * d + f) as usize];
                acc += diff * diff;
                f += 1u32;
            }
            if acc < best {
                best = acc;
                best_c = c;
            }
            c += 1u32;
        }
        label[i as usize] = best_c;
    }
}

/// Per-cluster sums of the batch over splits: `sums[s, c, f] = Σ x[f, i]` and `counts[s, c]`
/// (written by `f = 0`) over the batch points `i` of split `s` with `label[i] = c`. Splits keep each
/// sum short. One unit per `(s, c, f)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn batch_sums_kernel<F: Float>(
    x: &[F],
    label: &[u32],
    sums: &mut [F],
    counts: &mut [u32],
    n: u32,
    start: u32,
    len: u32,
    d: u32,
    k: u32,
    split_len: u32,
    splits: u32,
) {
    let unit = element_position() as u32;
    if unit < splits * k * d {
        let s = unit / (k * d);
        let rest = unit - s * k * d;
        let c = rest / d;
        let f = rest - c * d;
        let end = start + u32::min((s + 1u32) * split_len, len);
        let mut acc = F::new(0.0f32);
        let mut count = 0u32;
        let mut i = start + s * split_len;
        while i < end {
            if label[i as usize] == c {
                acc += x[(f * n + i) as usize];
                count += 1u32;
            }
            i += 1u32;
        }
        sums[unit as usize] = acc;
        if f == 0u32 {
            counts[(s * k + c) as usize] = count;
        }
    }
}

/// One batch's EMA update, on the device: with `n_c` and `s_c` the batch's count and sum of
/// cluster `c` (added over the splits), `mass ← (1 − α) mass + α n_c` and, when `n_c > 0`,
/// `centre ← centre + (α n_c / mass)(s_c / n_c − centre)`. In the last epoch (`close = 1`) the
/// batch's sums and counts are also added to `close_sums` / `close_counts`. One unit per cluster.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn ema_update_kernel<F: Float + CubeElement + LaunchArg>(
    sums: &[F],
    counts: &[u32],
    centres: &mut [F],
    mass: &mut [F],
    close_sums: &mut [F],
    close_counts: &mut [u32],
    k: u32,
    d: u32,
    splits: u32,
    alpha: F,
    close: u32,
) {
    let c = element_position() as u32;
    if c < k {
        let mut count = 0u32;
        let mut s = 0u32;
        while s < splits {
            count += counts[(s * k + c) as usize];
            s += 1u32;
        }
        let n_c = F::cast_from(count);
        let new_mass = (F::new(1.0f32) - alpha) * mass[c as usize] + alpha * n_c;
        if count > 0u32 {
            let w = alpha * n_c / new_mass;
            let mut f = 0u32;
            while f < d {
                let mut sum = F::new(0.0f32);
                let mut t = 0u32;
                while t < splits {
                    sum += sums[(t * k * d + c * d + f) as usize];
                    t += 1u32;
                }
                let at = (c * d + f) as usize;
                centres[at] += w * (sum / n_c - centres[at]);
                if close == 1u32 {
                    close_sums[at] += sum;
                }
                f += 1u32;
            }
        }
        mass[c as usize] = new_mass;
        if close == 1u32 {
            close_counts[c as usize] += count;
        }
    }
}

/// Closing: each centre becomes the mean of the points assigned to it in the last epoch (centres
/// without any keep their position). One unit per `(c, f)`.
#[cube(launch)]
pub fn close_centres_kernel<F: Float>(centres: &mut [F], close_sums: &[F], close_counts: &[u32], k: u32, d: u32) {
    let unit = element_position() as u32;
    if unit < k * d {
        let c = unit / d;
        let count = close_counts[c as usize];
        if count > 0u32 {
            centres[unit as usize] = close_sums[unit as usize] / F::cast_from(count);
        }
    }
}
