//! Memory-order conversion on the device, and channels gathered into a new order.
//!
//! A transpose is a copy from a transposed view, which cubecl-std already provides
//! (`cubecl::std::tensor::copy_into`, with a CPU-specific path): measured as fast as or faster than
//! a hand-written transpose kernel on both GPUs (`tests/bench_reduce.rs`).

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::zspace::{Shape, Strides};

use super::DspFloat;
use crate::core::buffer;

/// `output` (`[order.len(), samples]`) = the channels `order` of the channel-major `input`
/// (`[channels, samples]`), in that order: a channel remap on the device (any channel may be left
/// out or repeated).
///
/// # Panics
/// When a channel in `order` is not below `channels`.
pub fn execute_gather_channels<F: DspFloat>(client: &Client, input: &Handle, output: &Handle, channels: usize, samples: usize, order: &[usize]) {
    assert!(order.iter().all(|&c| c < channels), "gather_channels: a channel of {order:?} is not below {channels}");
    let total = order.len() * samples;
    if total == 0 {
        return;
    }
    let rows: Vec<u32> = order.iter().map(|&c| c as u32).collect();
    execute_gather_rows::<F>(client, input, &buffer::upload(client, &rows), output, channels, order.len(), samples);
}

/// [`execute_gather_channels`] with the order already on the device (`rows`: `out_channels` `u32`
/// indices, each below `channels`; not checked here), e.g. uploaded once for many windows.
pub fn execute_gather_rows<F: DspFloat>(client: &Client, input: &Handle, rows: &Handle, output: &Handle, channels: usize, out_channels: usize, samples: usize) {
    let total = out_channels * samples;
    if total == 0 {
        return;
    }
    let geom = dspk_core::compute::LaunchGeometry::elementwise(client, total);
    // SAFETY: the input holds `channels · samples` values and every index is below `channels` (the
    // caller's contract); the output holds `out_channels · samples`
    unsafe {
        super::kernels::gather_rows_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), channels * samples),
            BufferArg::from_raw_parts(rows.clone(), out_channels),
            BufferArg::from_raw_parts(output.clone(), total),
            samples as u32,
            total as u32,
        );
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn gathers_channels_in_the_given_order(client: &Client) {
        let (channels, samples) = (4usize, 1000usize);
        let data: Vec<f32> = (0..channels * samples).map(|i| i as f32).collect();
        let input = buffer::upload(client, &data);
        // Reordered, one left out, one repeated
        let order = [3usize, 0, 3];
        let output = buffer::empty::<f32>(client, order.len() * samples);
        execute_gather_channels::<f32>(client, &input, &output, channels, samples, &order);
        let got = buffer::download::<f32>(client, output);
        let expected: Vec<f32> = order.iter().flat_map(|&c| data[c * samples..(c + 1) * samples].to_vec()).collect();
        assert_eq!(&got[..expected.len()], &expected[..]);
    }
    runtime_test!(gather_channels_follows_the_order, gathers_channels_in_the_given_order);
}
