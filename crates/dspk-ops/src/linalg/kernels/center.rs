use cubecl::prelude::*;

/// `output[r, t] = input[offset + r · row_stride + t] − mean[r]` for `t < cols`: rows of a strided
/// buffer, centred, into a contiguous `[rows, cols]` buffer. The offset is applied in the kernel, so
/// the input binds at the buffer's start. One unit per output element (`ABSOLUTE_POS`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn center_rows_kernel<F: Float>(
    input: &[F],
    mean: &[F],
    output: &mut [F],
    offset: u32,
    row_stride: u32,
    cols: u32,
    total: u32,
) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        let r = e / cols;
        let t = e - r * cols;
        output[e as usize] = input[(offset + r * row_stride + t) as usize] - mean[r as usize];
    }
}
