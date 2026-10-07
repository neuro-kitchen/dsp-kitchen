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
    if rows == 0 || cols == 0 {
        return;
    }
    // SAFETY: both views cover exactly `rows · cols` elements of buffers that hold them
    let (input, output) = unsafe {
        (
            TensorBinding::from_raw_parts(input.clone(), Strides::new(&[1, cols]), Shape::new([cols, rows])),
            TensorBinding::from_raw_parts(output.clone(), Strides::new(&[rows, 1]), Shape::new([cols, rows])),
        )
    };
    cubecl::std::tensor::copy_into(client, input, output, F::elem_type_native());
}
