use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// Direct product of strided views, one unit per output element: `out[(b · m + i) · n + j] =
/// Σ_t lhs[l_off + b · l_bs + i · l_rs + t · l_cs] · rhs[r_off + b · r_bs + t · r_rs + j · r_cs]`
/// (`t < k`, summed in order). Output columns run along x ([`fn@sample_position`]), rows of all
/// batches along y ([`fn@channel_position`]): neighbouring units read neighbouring `rhs` columns, and
/// a row of `lhs` is shared by the plane. No reuse beyond that, so it suits products with little
/// to reuse (few rows, or a short inner dimension), where a tiled product's per-call cost and
/// unused tile space dominate. Offsets are applied here, so both inputs bind at their start.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn direct_matmul_kernel<F: Float>(
    lhs: &[F],
    rhs: &[F],
    out: &mut [F],
    batches: u32,
    m: u32,
    n: u32,
    k: u32,
    l_off: u32,
    l_bs: u32,
    l_rs: u32,
    l_cs: u32,
    r_off: u32,
    r_bs: u32,
    r_rs: u32,
    r_cs: u32,
) {
    let j = sample_position();
    let row = channel_position();
    if j < n && row < batches * m {
        let b = row / m;
        let i = row - b * m;
        let l_base = l_off + b * l_bs + i * l_rs;
        let r_base = r_off + b * r_bs + j * r_cs;
        let mut acc = F::new(0.0f32);
        let mut t = 0u32;
        while t < k {
            acc += lhs[(l_base + t * l_cs) as usize] * rhs[(r_base + t * r_rs) as usize];
            t += 1u32;
        }
        out[(row * n + j) as usize] = acc;
    }
}
