//! Memory-order conversion on the device.

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_core::compute::LaunchGeometry;

use super::DspFloat;

/// `output[c · rows + r] = input[r · cols + c]` (a `[rows, cols]` buffer to `[cols, rows]`). One unit
/// per output value, so writes are consecutive.
#[cube(launch)]
pub fn transpose_kernel<F: Float>(input: &[F], output: &mut [F], rows: u32, cols: u32) {
    let e = ABSOLUTE_POS as u32;
    if e < rows * cols {
        let c = e / rows;
        let r = e - c * rows;
        output[e as usize] = input[(r * cols + c) as usize];
    }
}

/// Transposes a `[rows, cols]` buffer of `F` into `output` (`[cols, rows]`), e.g. channel-major to
/// time-major.
pub fn transpose<F: DspFloat>(client: &Client, input: &Handle, output: &Handle, rows: usize, cols: usize) {
    let geom = LaunchGeometry::elementwise(client, rows * cols);
    unsafe {
        transpose_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), rows * cols),
            BufferArg::from_raw_parts(output.clone(), rows * cols),
            rows as u32,
            cols as u32,
        );
    }
}
