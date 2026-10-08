//! Kernels of the per-section clustering (see `kilosort4/clustering.rs`). Section points are
//! feature-major (`x[f · n + i]`, `f = slot · n_pcs + p`); `side[label]` says which half of a
//! merge-tree node a cluster is on: [`SIDE_NONE`], [`SIDE_NEG`] (`y = −1`) or [`SIDE_POS`] (`y = +1`).

use cubecl::prelude::*;
use dsp_base::core::DspFloat;

// Kernels compare against the literals 0 / 1 / 2 (see `weigh_node_kernel`)
pub const SIDE_NONE: u32 = 0;
pub const SIDE_NEG: u32 = 1;
pub const SIDE_POS: u32 = 2;

/// Section embedding: spike `i`'s features on its centre's `nc` channels (`feat[(i · nc + c) ·
/// np + p]`) go to the section's channel slots (`slots[i · nc + c]`): `x[(slot · np + p) · n + i]`
/// (`x` zeroed: channels a spike does not cover stay 0). One unit per `(i, c, p)`.
#[cube(launch)]
pub fn embed_section_kernel<F: Float>(feat: &[F], slots: &[u32], x: &mut [F], n: u32, nc: u32, np: u32) {
    let unit = ABSOLUTE_POS as u32;
    if unit < n * nc * np {
        let i = unit / (nc * np);
        let rest = unit - i * nc * np;
        let c = rest / np;
        let p = rest - c * np;
        let slot = slots[(i * nc + c) as usize];
        x[((slot * np + p) * n + i) as usize] = feat[unit as usize];
    }
}

/// Weighted regression inputs of one merge-tree node: `xs[f, i] = √w_i · x[f, i]`, `v[i] = √w_i · y_i`
/// and `s[i] = √w_i` for the node's points (`w`, `y` by side), 0 for the others, so that
/// `xs · xsᵀ = Σ w x xᵀ`, `xs · v = Σ w y x` and `xs · s = Σ w x`. One unit per point.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn weigh_node_kernel<F: DspFloat>(
    x: &[F],
    labels: &[u32],
    side: &[u32],
    xs: &mut [F],
    v: &mut [F],
    s_out: &mut [F],
    n: u32,
    d: u32,
    c: u32,
    root_w_neg: F,
    root_w_pos: F,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let label = labels[i as usize];
        // From a literal: a mutable variable started from a constant is compile-time in CubeCL
        let mut s = 0u32;
        if label < c {
            s = side[label as usize];
        }
        let mut scale = F::new(0.0f32);
        let mut y = F::new(0.0f32);
        if s == 1u32 {
            scale = root_w_neg;
            y = -root_w_neg;
        }
        if s == 2u32 {
            scale = root_w_pos;
            y = root_w_pos;
        }
        let mut f = 0u32;
        while f < d {
            xs[(f * n + i) as usize] = scale * x[(f * n + i) as usize];
            f += 1u32;
        }
        v[i as usize] = y;
        s_out[i as usize] = scale;
    }
}

/// Histogram of the node's projections `u · x_i + bias` in `bins` equal bins over `[lo, hi]` (`hi` in
/// the last bin; values outside are left out); `hist` zeroed. One unit per point.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn projection_histogram_kernel<F: DspFloat>(
    x: &[F],
    labels: &[u32],
    side: &[u32],
    u: &[F],
    hist: &mut [Atomic<u32>],
    n: u32,
    d: u32,
    c: u32,
    bins: u32,
    bias: F,
    lo: F,
    hi: F,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let label = labels[i as usize];
        if label < c {
            if side[label as usize] != 0u32 {
                let mut p = bias;
                let mut f = 0u32;
                while f < d {
                    p += u[f as usize] * x[(f * n + i) as usize];
                    f += 1u32;
                }
                if p >= lo && p <= hi {
                    let scaled = (p - lo) / (hi - lo) * F::cast_from(bins);
                    let b = u32::min(u32::cast_from(F::floor(scaled)), bins - 1u32);
                    Atomic::fetch_add(&hist[b as usize], 1u32);
                }
            }
        }
    }
}
