use cubecl::prelude::*;
use dspk_core::compute::element_position;

/// `x[e] −= mean[e mod cols]`: rows of a contiguous `[rows, cols]` buffer centred on the column
/// means, in place. One unit per element ([`fn@element_position`]).
#[cube(launch)]
pub fn centre_columns_kernel<F: Float>(x: &mut [F], mean: &[F], cols: u32, total: u32) {
    let e = element_position() as u32;
    if e < total {
        x[e as usize] = x[e as usize] - mean[(e % cols) as usize];
    }
}

/// `acc[e] += x[e]`. One unit per element ([`fn@element_position`]).
#[cube(launch)]
pub fn add_assign_kernel<F: Float>(acc: &mut [F], x: &[F], total: u32) {
    let e = element_position() as u32;
    if e < total {
        acc[e as usize] = acc[e as usize] + x[e as usize];
    }
}

