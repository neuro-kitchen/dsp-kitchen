use cubecl::prelude::*;
use dspk_core::compute::element_position;

/// Samples `offset .. offset + len` of every row of a channel-major buffer whose rows are
/// `in_row_len` long, into a compact `[rows, len]` buffer.
#[cube(launch)]
pub fn crop_rows_kernel<F: Float>(input: &[F], output: &mut [F], in_row_len: u32, offset: u32, len: u32, total: u32) {
    let i = element_position() as u32;
    if i < total {
        let row = i / len;
        let t = i % len;
        output[i as usize] = input[(row * in_row_len + offset + t) as usize];
    }
}
