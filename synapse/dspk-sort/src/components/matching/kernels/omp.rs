//! Kernels of [`crate::components::matching::omp`].

use cubecl::prelude::*;
use dspk_core::compute::element_position;

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
    let e = element_position() as u32;
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
