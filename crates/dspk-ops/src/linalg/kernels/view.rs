use cubecl::prelude::*;

/// Copies a strided `[batches, rows, cols]` view into a contiguous buffer: `output[(b · rows + r) ·
/// cols + c] = input[offset + b · batch_stride + r · row_stride + c · col_stride]`. The offset is
/// applied in the kernel, so the input binds at the buffer's start (device offset alignment does
/// not apply). One unit per output element (`ABSOLUTE_POS`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gather_view_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    offset: u32,
    rows: u32,
    cols: u32,
    batch_stride: u32,
    row_stride: u32,
    col_stride: u32,
    total: u32,
) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        let c = e % cols;
        let br = e / cols;
        let r = br % rows;
        let b = br / rows;
        output[e as usize] = input[(offset + b * batch_stride + r * row_stride + c * col_stride) as usize];
    }
}
