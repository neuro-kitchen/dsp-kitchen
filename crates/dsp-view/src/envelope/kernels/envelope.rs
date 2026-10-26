use cubecl::prelude::*;
use dsp_core::compute::{negative_infinity, positive_infinity, row_position};

/// Values stored per column: `min`, then `max`.
pub const PAIR: u32 = 2;

/// Min/max of samples `edges[x]..edges[x + 1]` of every channel row (rows `row_stride` apart),
/// into `out[PAIR · (channel · columns + x)]` (min) and the value after it (max). Columns with no
/// sample other than NaN stay `[+∞, −∞]`. `units` is the cube's x size (a power of two).
#[cube(launch)]
pub fn envelope_kernel<F: Float>(
    input: &[F],
    edges: &[u32],
    out: &mut [F],
    columns: u32,
    rows: u32,
    row_stride: u32,
    #[comptime] units: u32,
) {
    let row = row_position();
    // `row` is uniform across the cube, so every unit of a cube takes the same branch
    if row < rows {
        let unit = UNIT_POS_X;
        let channel = row / columns;
        let column = row % columns;
        let base = channel * row_stride;
        let end = edges[(column + 1u32) as usize];

        // Comparisons are false for NaN, so NaN samples never replace a bound
        let mut lo = positive_infinity::<F>();
        let mut hi = negative_infinity::<F>();
        let mut s = edges[column as usize] + unit;
        while s < end {
            let x = input[(base + s) as usize];
            if x < lo {
                lo = x;
            }
            if x > hi {
                hi = x;
            }
            s += units;
        }

        let mut lo_s = Shared::<[F]>::new_slice(comptime!(units as usize));
        let mut hi_s = Shared::<[F]>::new_slice(comptime!(units as usize));
        lo_s[unit as usize] = lo;
        hi_s[unit as usize] = hi;
        sync_cube();

        let stride = RuntimeCell::<u32>::new(units / 2u32);
        while stride.read() > 0u32 {
            let s = stride.read();
            if unit < s {
                let other = (unit + s) as usize;
                let (lo_o, hi_o) = (lo_s[other], hi_s[other]);
                if lo_o < lo_s[unit as usize] {
                    lo_s[unit as usize] = lo_o;
                }
                if hi_o > hi_s[unit as usize] {
                    hi_s[unit as usize] = hi_o;
                }
            }
            sync_cube();
            stride.store(s / 2u32);
        }

        if unit == 0u32 {
            out[(row * PAIR) as usize] = lo_s[0];
            out[(row * PAIR + 1u32) as usize] = hi_s[0];
        }
    }
}
