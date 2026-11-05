//! Memory-order conversion on the device.
//!
//! A transpose is a copy from a transposed view, which cubecl-std already provides
//! (`cubecl::std::tensor::copy_into`, with a CPU-specific path): measured as fast as or faster than
//! a hand-written transpose kernel on both GPUs (`tests/bench_reduce.rs`).

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::zspace::{Shape, Strides};

use super::DspFloat;

/// Transposes a `[rows, cols]` buffer of `F` into `output` (`[cols, rows]`), e.g. channel-major to
/// time-major.
pub fn transpose<F: DspFloat>(client: &Client, input: &Handle, output: &Handle, rows: usize, cols: usize) {
    transpose_strided::<F>(client, input, output, rows, cols, cols, rows);
}

/// [`transpose`] with row strides: element `(r, c)` of the input is `input[r · in_row_stride + c]`,
/// and lands at `output[c · out_row_stride + r]` (`in_row_stride ≥ cols`, `out_row_stride ≥ rows`;
/// the padding of either buffer is neither read nor written).
pub fn transpose_strided<F: DspFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    rows: usize,
    cols: usize,
    in_row_stride: usize,
    out_row_stride: usize,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    debug_assert!(in_row_stride >= cols && out_row_stride >= rows, "row strides below the row lengths");
    if in_row_stride != cols || out_row_stride != rows {
        // Padded rows: a plain gather kernel. cubecl-std's `copy_into` gave wrong values on the
        // CPU runtime with a padded output (its CPU path for strided inputs; `filter_blocks`)
        let geom = dspk_core::compute::LaunchGeometry::elementwise(client, rows * cols);
        // SAFETY: the input holds `rows · in_row_stride` values (its last row may stop at `cols`),
        // the output `cols · out_row_stride`
        unsafe {
            super::kernels::transpose_strided_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(input.clone(), (rows - 1) * in_row_stride + cols),
                BufferArg::from_raw_parts(output.clone(), (cols - 1) * out_row_stride + rows),
                rows as u32,
                cols as u32,
                in_row_stride as u32,
                out_row_stride as u32,
            );
        }
        return;
    }
    // SAFETY: both views cover exactly `rows · cols` elements of buffers that hold them
    let (input, output) = unsafe {
        (
            TensorBinding::from_raw_parts(input.clone(), Strides::new(&[1, in_row_stride]), Shape::new([cols, rows])),
            TensorBinding::from_raw_parts(output.clone(), Strides::new(&[out_row_stride, 1]), Shape::new([cols, rows])),
        )
    };
    cubecl::std::tensor::copy_into(client, input, output, F::elem_type_native());
}
