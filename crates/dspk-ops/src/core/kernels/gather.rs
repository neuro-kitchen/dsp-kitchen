//! Rows gathered by index (channels of a channel-major buffer, vectors of a matrix).

use cubecl::prelude::*;
use dspk_core::compute::element_position;

/// `out[r, f] = src[rows[r], f]`: rows of a row-major `[_, dim]` buffer gathered by index. One unit
/// per output element ([`fn@element_position`]).
#[cube(launch)]
pub fn gather_rows_kernel<F: Float>(src: &[F], rows: &[u32], out: &mut [F], dim: u32, total: u32) {
    let e = element_position() as u32;
    if e < total {
        let r = e / dim;
        let f = e - r * dim;
        out[e as usize] = src[(rows[r as usize] * dim + f) as usize];
    }
}
