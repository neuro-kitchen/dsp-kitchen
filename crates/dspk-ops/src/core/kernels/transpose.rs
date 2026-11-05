//! A transpose between buffers with padded rows.

use cubecl::prelude::*;
use dspk_core::compute::element_position;

/// One unit per element `(r, c)` of a `[rows, cols]` input whose rows are `in_row_stride` apart:
/// `output[c · out_row_stride + r] = input[r · in_row_stride + c]`. The padding of either buffer
/// is neither read nor written.
#[cube(launch)]
pub fn transpose_strided_kernel<F: Float>(input: &[F], output: &mut [F], rows: u32, cols: u32, in_row_stride: u32, out_row_stride: u32) {
    let e = element_position() as u32;
    if e < rows * cols {
        let r = e / cols;
        let c = e % cols;
        output[(c * out_row_stride + r) as usize] = input[(r * in_row_stride + c) as usize];
    }
}
