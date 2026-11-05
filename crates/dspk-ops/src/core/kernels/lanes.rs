//! Vectors of `N` consecutive samples that start anywhere in a row, for vectorized kernels on rows
//! whose length is not a multiple of `N`.
//!
//! An aligned kernel reads and writes `&[Vector<F, N>]`: row `r` starts at vector `r · len / N`,
//! which only exists when `N` divides `len` (385 channels, 66 234-sample windows). The unaligned
//! variant of each kernel keeps the same arithmetic on `Vector<F, N>` but gathers its inputs and
//! scatters its outputs lane by lane with these helpers: the last vector of a row is partly past its
//! end, and only its `valid` lanes are stored. Lanes are independent, so the stored lanes are
//! bit-identical to the scalar kernels.

use cubecl::prelude::*;

/// `input[start + l]` in lane `l` for `l < valid`; lanes from `valid` on repeat lane `valid − 1`
/// (in bounds, and never stored by [`fn@store_lanes`]). `valid` ≥ 1.
#[cube]
pub fn load_lanes<F: Float, N: Size>(input: &[F], start: usize, valid: u32) -> Vector<F, N> {
    let mut v = Vector::<F, N>::empty();
    let n = v.vector_size();
    let last = valid - 1u32;
    #[unroll]
    for l in 0..n {
        v.insert(l, input[start + u32::min(l as u32, last) as usize]);
    }
    v
}

/// `output[start + l] = value[l]` for the lanes `l < valid`.
#[cube]
pub fn store_lanes<F: Float, N: Size>(output: &mut [F], start: usize, value: Vector<F, N>, valid: u32) {
    let n = value.vector_size();
    #[unroll]
    for l in 0..n {
        if (l as u32) < valid {
            output[start + l] = value.extract(l);
        }
    }
}
