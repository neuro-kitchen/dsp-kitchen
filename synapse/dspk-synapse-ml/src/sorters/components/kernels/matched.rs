//! Kernels of [`crate::sorters::components::matched`].

use cubecl::prelude::*;

/// One unit per element `(c, i)` of `out` (`[channels, samples]`): `Σₖ proto[k] · x[c, i + k]`
/// for `i + len ≤ samples` (the correlation's valid part), 0 past it.
#[cube(launch)]
pub fn correlate_prototype_kernel<F: Float>(x: &[F], proto: &[F], out: &mut [F], samples: u32, len: u32, total: u32) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        let c = e / samples;
        let i = e - c * samples;
        let mut acc = F::new(0.0f32);
        if i + len <= samples {
            let base = c * samples + i;
            let mut k = 0u32;
            while k < len {
                acc += proto[k as usize] * x[(base + k) as usize];
                k += 1u32;
            }
        }
        out[e as usize] = acc;
    }
}

/// One unit per element `(r, i)` of `out` (`[rows, samples]`): `Σ w · corr[c, i]` over row `r`'s
/// channels (CSR: `offsets[r]..offsets[r + 1]` of `cols` / `weights`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn spatial_sum_kernel<F: Float>(
    corr: &[F],
    offsets: &[u32],
    cols: &[u32],
    weights: &[F],
    out: &mut [F],
    samples: u32,
    total: u32,
) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        let r = e / samples;
        let i = e - r * samples;
        let mut acc = F::new(0.0f32);
        let mut k = offsets[r as usize];
        let end = offsets[(r + 1u32) as usize];
        while k < end {
            acc += weights[k as usize] * corr[(cols[k as usize] * samples + i) as usize];
            k += 1u32;
        }
        out[e as usize] = acc;
    }
}

/// `out[i] = x[channels[i] · samples + at[i]]`: values of a `[channels, samples]` buffer at points.
#[cube(launch)]
pub fn gather_points_kernel<F: Float>(x: &[F], channels: &[u32], at: &[u32], out: &mut [F], samples: u32, n: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        out[i as usize] = x[(channels[i as usize] * samples + at[i as usize]) as usize];
    }
}

/// One unit per element `(t, i)` of `out` (`[templates, peaks]`, `peaks = samples − width + 1`):
/// `Σ_r Σ_k temporal[t, k, r] · y[t · rank + r, i + k]`, with `y` the spatially filtered and
/// scaled data (`[templates · rank, samples]`) and `temporal` `[templates, width, rank]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn omp_scalar_products_kernel<F: Float>(
    y: &[F],
    temporal: &[F],
    out: &mut [F],
    samples: u32,
    width: u32,
    rank: u32,
    peaks: u32,
    total: u32,
) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        let t = e / peaks;
        let i = e - t * peaks;
        let mut acc = F::new(0.0f32);
        let mut r = 0u32;
        while r < rank {
            let row = (t * rank + r) * samples + i;
            let mut k = 0u32;
            while k < width {
                acc += temporal[((t * width + k) * rank + r) as usize] * y[(row + k) as usize];
                k += 1u32;
            }
            r += 1u32;
        }
        out[e as usize] = acc;
    }
}
