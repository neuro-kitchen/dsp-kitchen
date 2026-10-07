//! Device reads past either end of a row ([`crate::core::EdgeMode`] ids).

use cubecl::prelude::*;

use crate::core::edge::{EDGE_NEAREST, EDGE_ODD, EDGE_REFLECT};

/// Sample `j − pad` of the row starting at `input[base]` with `len` samples, extended by `mode`
/// (`mode` is an [`EdgeMode::id`]). `j` runs over `0..len + 2·pad`. Reflections are one row deep:
/// positions further out than `len − 1` samples read the far end of the row (never out of bounds).
#[cube]
pub fn read_extended<F: Float>(input: &[F], base: usize, len: u32, pad: u32, j: u32, #[comptime] mode: u32) -> F {
    read_extended_strided::<F>(input, base, 1u32, len, pad, j, mode)
}

/// [`read_extended`] for a row whose consecutive samples are `stride` values apart (sample `i` at
/// `input[base + i · stride]`), e.g. a channel of a time-major buffer.
#[cube]
pub fn read_extended_strided<F: Float>(input: &[F], base: usize, stride: u32, len: u32, pad: u32, j: u32, #[comptime] mode: u32) -> F {
    let last = len - 1u32;
    let mut value = F::new(0.0f32);
    if j < pad {
        // Left of the row: distance k ≥ 1 from sample 0
        let k = pad - j;
        if comptime!(mode == EDGE_ODD) {
            value = F::new(2.0f32) * input[base] - input[base + (u32::min(k, last) * stride) as usize];
        } else if comptime!(mode == EDGE_REFLECT) {
            value = input[base + (u32::min(k - 1u32, last) * stride) as usize];
        } else if comptime!(mode == EDGE_NEAREST) {
            value = input[base];
        }
    } else if j - pad <= last {
        value = input[base + ((j - pad) * stride) as usize];
    } else {
        // Right of the row: distance k ≥ 1 from the last sample
        let k = u32::min(j - pad - last, last + 1u32);
        if comptime!(mode == EDGE_ODD) {
            value = F::new(2.0f32) * input[base + (last * stride) as usize] - input[base + ((last - u32::min(k, last)) * stride) as usize];
        } else if comptime!(mode == EDGE_REFLECT) {
            value = input[base + ((last + 1u32 - k) * stride) as usize];
        } else if comptime!(mode == EDGE_NEAREST) {
            value = input[base + (last * stride) as usize];
        }
    }
    value
}
