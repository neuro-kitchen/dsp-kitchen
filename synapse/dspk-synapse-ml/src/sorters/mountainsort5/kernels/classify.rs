//! Kernels of [`crate::sorters::mountainsort5::scheme2`]: the per-channel classifiers.
//!
//! Classifier tables, concatenated over channels: channel `c` has `n_mask[c]` mask channels,
//! `n_comp[c]` components and `n_train[c]` training points; its mean (`T · n_mask` values) starts
//! at `mean_off[c]`, its components (`[T · n_mask, n_comp]`) at `comp_off[c]`, its training
//! features (`[n_train, n_comp]`) at `feat_off[c]`.

use cubecl::prelude::*;

/// One unit per `(event e, component k)` with `k < max_comp`: the projection of event `e`'s masked
/// snippet (`snippets`, `[events, width, max_neighbours]`) on component `k` of its channel's
/// classifier, centred on its mean, into `out[e · max_comp + k]` (0 past the channel's count).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn project_classifier_kernel<F: Float>(
    snippets: &[F],
    ev_channels: &[u32],
    n_mask: &[u32],
    n_comp: &[u32],
    mean_off: &[u32],
    comp_off: &[u32],
    means: &[F],
    components: &[F],
    out: &mut [F],
    total: u32,
    width: u32,
    max_neighbours: u32,
    max_comp: u32,
) {
    let e2 = ABSOLUTE_POS as u32;
    if e2 < total {
        let k = e2 % max_comp;
        let e = e2 / max_comp;
        let ch = ev_channels[e as usize];
        let nm = n_mask[ch as usize];
        let nc = n_comp[ch as usize];
        let mo = mean_off[ch as usize];
        let co = comp_off[ch as usize];
        let mut y = F::new(0.0f32);
        if k < nc {
            let mut s = 0u32;
            while s < width {
                let mut j = 0u32;
                while j < nm {
                    let f = s * nm + j;
                    let x = snippets[((e * width + s) * max_neighbours + j) as usize] - means[(mo + f) as usize];
                    y += x * components[(co + f * nc + k) as usize];
                    j += 1u32;
                }
                s += 1u32;
            }
        }
        out[e2 as usize] = y;
    }
}

/// One unit per event: the index (within its channel's training points) of its **second** nearest
/// training point by Euclidean distance on the channel's components (the first may be the event
/// itself), ties to the lower index; the nearest when there is only one.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn second_nearest_kernel<F: Float>(
    projections: &[F],
    ev_channels: &[u32],
    n_comp: &[u32],
    n_train: &[u32],
    feat_off: &[u32],
    features: &[F],
    out: &mut [u32],
    n_events: u32,
    max_comp: u32,
) {
    let e = ABSOLUTE_POS as u32;
    if e < n_events {
        let ch = ev_channels[e as usize];
        let nc = n_comp[ch as usize];
        let nt = n_train[ch as usize];
        let fo = feat_off[ch as usize];
        let mut d1 = F::new(-1.0f32);
        let mut d2 = F::new(-1.0f32);
        let mut j1 = 0u32;
        let mut j2 = 0u32;
        let mut j = 0u32;
        while j < nt {
            let mut d = F::new(0.0f32);
            let mut k = 0u32;
            while k < nc {
                let diff = projections[(e * max_comp + k) as usize] - features[(fo + j * nc + k) as usize];
                d += diff * diff;
                k += 1u32;
            }
            let first = d1 < F::new(0.0f32) || d < d1;
            if first {
                d2 = d1;
                j2 = j1;
                d1 = d;
                j1 = j;
            }
            if !first && (d2 < F::new(0.0f32) || d < d2) {
                d2 = d;
                j2 = j;
            }
            j += 1u32;
        }
        let mut pick = j2;
        if nt < 2u32 {
            pick = j1;
        }
        out[e as usize] = pick;
    }
}
