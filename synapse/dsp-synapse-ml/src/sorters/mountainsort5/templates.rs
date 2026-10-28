//! MountainSort 5 templates and alignment (`core/compute_templates.py`, `sorting_scheme1.py`).
//!
//! - [`median_templates`]: each cluster's **median** snippet in upstream's dense form (`[T, M]`,
//!   the masked-out channels of a member count as zeros, as in upstream's dense snippets).
//! - [`align_templates`]: each template pair's best circular shift (the largest inner product of
//!   `roll(t₁, o)` with `t₂`, `o > T/2` read as `o − T`), then per-template offsets: the
//!   inner-product-weighted average of `pairwise(k₁, k₂) + offset(k₂)` over the others, truncated
//!   toward zero (`int(·)`), iterated until nothing changes (at most 20 passes). The `T` products
//!   of every pair run on the device (one `[K, T·M] × [T·M, K]` product per shift).
//! - [`offsets_to_peak`]: the sample of the template's `detect_sign` peak on its peak channel,
//!   minus `T1`.
//! - [`peak_channels`]: the channel of each template's minimum (upstream's
//!   `argmin(min(template, axis=time))`, whatever the sign).

use cubecl::prelude::*;
use dsp_base::core::buffer;
use dsp_base::linalg::{matmul, MatrixView};

use super::snippets::MaskedSnippets;

/// Median snippets of clusters `1..=k` (`labels[i]` of event `i`; 0 is no cluster), `[k, T, M]`.
/// An empty cluster's template is zeros.
///
/// # Panics
///
/// If `labels` has another length than the snippets.
pub fn median_templates(snippets: &MaskedSnippets, labels: &[u32], k: usize) -> Vec<f32> {
    assert_eq!(labels.len(), snippets.len(), "one label per snippet");
    let (w, m, nb) = (snippets.width, snippets.channels, snippets.mask.max_neighbours);
    let mut out = vec![0.0f32; k * w * m];
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &l) in labels.iter().enumerate() {
        if l >= 1 && (l as usize) <= k {
            members[l as usize - 1].push(i);
        }
    }
    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); w * m];
    for (c, idx) in members.iter().enumerate() {
        if idx.is_empty() {
            continue;
        }
        buckets.iter_mut().for_each(Vec::clear);
        for &i in idx {
            let row = snippets.row(i);
            for (slot, ch) in snippets.mask.of(snippets.event_channels[i] as usize).enumerate() {
                for s in 0..w {
                    buckets[s * m + ch].push(row[s * nb + slot]);
                }
            }
        }
        for (e, bucket) in buckets.iter_mut().enumerate() {
            out[c * w * m + e] = median_with_zeros(bucket, idx.len());
        }
    }
    out
}

/// NumPy's median of `values` plus `n − values.len()` zeros.
fn median_with_zeros(values: &mut [f32], n: usize) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let zeros = n - values.len();
    // The sorted sequence with the zeros inserted where they belong
    let below = values.partition_point(|&v| v < 0.0);
    let at = |r: usize| -> f32 {
        if r < below {
            values[r]
        } else if r < below + zeros {
            0.0
        } else {
            values[r - zeros]
        }
    };
    if n % 2 == 1 { at(n / 2) } else { 0.5 * (at(n / 2 - 1) + at(n / 2)) }
}

/// Offsets of every template (`[k, T, M]`; module docs).
pub fn align_templates(client: &Client, templates: &[f32], k: usize, t: usize, m: usize) -> Vec<i32> {
    if k == 0 {
        return Vec::new();
    }
    let d = t * m;
    let base = buffer::upload(client, templates);
    let base_t = MatrixView::row_major(&base, k * d, k, d).transposed();
    let products = buffer::empty::<f32>(client, k * k);
    // best[(k1, k2)] = (inner product, offset)
    let mut best = vec![(f32::NEG_INFINITY, 0i32); k * k];
    let mut rolled = vec![0.0f32; k * d];
    for o in 0..t {
        // roll(t₁, o): sample s takes sample s − o
        for c in 0..k {
            let src = &templates[c * d..(c + 1) * d];
            let dst = &mut rolled[c * d..(c + 1) * d];
            for s in 0..t {
                let from = (s + t - o) % t;
                dst[s * m..(s + 1) * m].copy_from_slice(&src[from * m..(from + 1) * m]);
            }
        }
        let r = buffer::upload(client, &rolled);
        matmul::<f32>(client, &MatrixView::row_major(&r, k * d, k, d), &base_t, &products, k * k);
        for (e, ip) in buffer::download::<f32>(client, products.clone()).into_iter().enumerate() {
            if ip > best[e].0 {
                best[e] = (ip, o as i32);
            }
        }
    }
    let half = (t / 2) as i32;
    let pairwise: Vec<i32> = best.iter().map(|&(_, o)| if o > half { o - t as i32 } else { o }).collect();
    let mut offsets = vec![0i32; k];
    for _ in 0..20 {
        let mut changed = false;
        for k1 in 0..k {
            let (mut sum, mut weight) = (0.0f64, 0.0f64);
            for k2 in 0..k {
                if k1 != k2 {
                    let w = best[k1 * k + k2].0 as f64;
                    sum += w * (pairwise[k1 * k + k2] + offsets[k2]) as f64;
                    weight += w;
                }
            }
            let avg = if weight > 0.0 { (sum / weight).trunc() as i32 } else { 0 };
            if avg != offsets[k1] {
                changed = true;
                offsets[k1] = avg;
            }
        }
        if !changed {
            break;
        }
    }
    offsets
}

/// `|template|` in the sense of `sign` (−1: `−x`, +1: `x`, 0: `|x|`).
fn signed(x: f32, sign: i8) -> f32 {
    match sign {
        s if s < 0 => -x,
        s if s > 0 => x,
        _ => x.abs(),
    }
}

/// First index of the largest value.
fn argmax(v: impl Iterator<Item = f32>) -> usize {
    let mut best = (0, f32::NEG_INFINITY);
    for (i, x) in v.enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    best.0
}

/// Offset of each template's peak from `n_before` (module docs).
pub fn offsets_to_peak(templates: &[f32], k: usize, t: usize, m: usize, sign: i8, n_before: usize) -> Vec<i32> {
    (0..k)
        .map(|c| {
            let tp = &templates[c * t * m..(c + 1) * t * m];
            let channel = argmax((0..m).map(|ch| (0..t).map(|s| signed(tp[s * m + ch], sign)).fold(f32::NEG_INFINITY, f32::max)));
            argmax((0..t).map(|s| signed(tp[s * m + channel], sign))) as i32 - n_before as i32
        })
        .collect()
}

/// Channel of each template's minimum (module docs).
pub fn peak_channels(templates: &[f32], k: usize, t: usize, m: usize) -> Vec<usize> {
    (0..k)
        .map(|c| {
            let tp = &templates[c * t * m..(c + 1) * t * m];
            argmax((0..m).map(|ch| -(0..t).map(|s| tp[s * m + ch]).fold(f32::INFINITY, f32::min)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;

    #[test]
    fn medians_count_masked_channels_as_zeros() {
        assert_eq!(median_with_zeros(&mut [3.0, -1.0], 3), 0.0);
        assert_eq!(median_with_zeros(&mut [3.0, 5.0], 3), 3.0);
        assert_eq!(median_with_zeros(&mut [-3.0, -5.0, -1.0], 4), -2.0);
        assert_eq!(median_with_zeros(&mut [2.0, 4.0], 2), 3.0);
    }

    #[test]
    fn peaks_and_offsets() {
        // One template, 2 channels, 5 samples: minimum −9 on channel 1 at sample 3
        let tp = vec![0.0, 0.0, -1.0, 0.0, -2.0, -3.0, -4.0, -9.0, 0.0, 1.0];
        assert_eq!(peak_channels(&tp, 1, 5, 2), vec![1]);
        assert_eq!(offsets_to_peak(&tp, 1, 5, 2, -1, 2), vec![1]);
        assert_eq!(offsets_to_peak(&tp, 1, 5, 2, 1, 2), vec![2], "positive peak: 1.0 at sample 4 of channel 1");
    }

    /// Two copies of one waveform, the second shifted by 3 samples: their offsets differ by 3.
    #[test]
    fn alignment_finds_the_shift() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (t, m) = (20usize, 2usize);
        let wave = |s: i32| (-((s as f32 - 8.0) / 2.0).powi(2)).exp() * -10.0;
        let mut tp = vec![0.0f32; 2 * t * m];
        for s in 0..t {
            tp[s * m] = wave(s as i32);
            tp[t * m + s * m] = wave(s as i32 - 3);
        }
        let off = align_templates(&client, &tp, 2, t, m);
        // Template 1 is template 0 rolled by 3: aligning means offsets differ by −3 / +3
        assert_eq!(off[0] - off[1], 3, "{off:?}");
    }
}
