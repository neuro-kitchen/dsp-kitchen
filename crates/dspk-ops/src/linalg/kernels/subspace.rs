use cubecl::prelude::*;

/// `x[e] −= mean[e mod cols]`: rows of a contiguous `[rows, cols]` buffer centred on the column
/// means, in place. One unit per element (`ABSOLUTE_POS`).
#[cube(launch)]
pub fn centre_columns_kernel<F: Float>(x: &mut [F], mean: &[F], cols: u32, total: u32) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        x[e as usize] = x[e as usize] - mean[(e % cols) as usize];
    }
}

/// `acc[e] += x[e]`. One unit per element (`ABSOLUTE_POS`).
#[cube(launch)]
pub fn add_assign_kernel<F: Float>(acc: &mut [F], x: &[F], total: u32) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        acc[e as usize] = acc[e as usize] + x[e as usize];
    }
}

/// `out[r, f] = src[rows[r], f]`: rows of a row-major `[_, dim]` buffer gathered by index. One unit
/// per output element (`ABSOLUTE_POS`).
#[cube(launch)]
pub fn gather_rows_kernel<F: Float>(src: &[F], rows: &[u32], out: &mut [F], dim: u32, total: u32) {
    let e = ABSOLUTE_POS as u32;
    if e < total {
        let r = e / dim;
        let f = e - r * dim;
        out[e as usize] = src[(rows[r as usize] * dim + f) as usize];
    }
}
