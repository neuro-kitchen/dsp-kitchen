use cubecl::prelude::*;
use dsp_core::compute::row_position;

/// Bits of one radix digit of [`row_abs_kth_radix_kernel`].
pub const RADIX_BITS: u32 = 8;

/// Bins of a digit's histogram.
pub const RADIX_BINS: u32 = 1 << RADIX_BITS;

/// One cube per row: the `k`-th smallest (0-based) `|x|` among columns `col_start..col_start + cols`
/// of row `row` (rows `row_stride` apart), exactly.
///
/// The bits of a non-negative IEEE float, read as an unsigned integer `K` of the same width, sort
/// like its value (subnormals and zero included; NaN above +∞). So the `k`-th smallest `|x|` is the
/// `k`-th smallest key, found digit by digit from the most significant ([`RADIX_BITS`] each): a
/// pass counts, in a shared-memory histogram, the keys that agree with the digits fixed so far,
/// and one unit walks the histogram to the bin holding rank `k`, fixes that digit and subtracts the
/// keys in lower bins from `k`. After `key_bits / RADIX_BITS` passes over the row the key is the
/// exact bit pattern of the answer. `key_bits` is the width of `F` and `K` (from the host:
/// `K::size_bits()` inside a kernel does not give it); `units` is the cube's x size.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn row_abs_kth_radix_kernel<F: Float, K: Int>(
    input: &[F],
    out: &mut [F],
    rows: u32,
    row_stride: u32,
    col_start: u32,
    cols: u32,
    k: u32,
    #[comptime] key_bits: u32,
    #[comptime] units: u32,
) {
    let row = row_position();
    // `row` is uniform across the cube, so every unit of a cube takes the same branch
    if row < rows {
        let unit = UNIT_POS_X;
        let base = row * row_stride + col_start;
        let passes = comptime!(key_bits / RADIX_BITS);
        let digit_mask = K::cast_from(RADIX_BINS - 1);
        // Atomics are reached through a reference (`Atomic::fetch_add(&hist[i], ..)`): calling the
        // method on the indexed element works on a copy of it in CubeCL 0.11
        let hist = Shared::<[Atomic<u32>]>::new_slice(comptime!(RADIX_BINS as usize));
        // [prefix, mask] of the digits fixed so far, and the rank left to find
        let mut fixed = Shared::<[K]>::new_slice(2usize);
        let mut rank = Shared::<[u32]>::new_slice(1usize);
        if unit == 0u32 {
            fixed[0] = K::cast_from(0u32);
            fixed[1] = K::cast_from(0u32);
            rank[0] = k;
        }
        sync_cube();

        #[unroll]
        for pass in 0..passes {
            let shift = K::cast_from(comptime!(key_bits - RADIX_BITS * (pass + 1)));
            let mut b = unit;
            while b < RADIX_BINS {
                Atomic::store(&hist[b as usize], 0u32);
                b += units;
            }
            sync_cube();
            let (prefix, mask) = (fixed[0], fixed[1]);
            let mut col = unit;
            while col < cols {
                let key = K::reinterpret(F::abs(input[(base + col) as usize]));
                if (key & mask) == prefix {
                    let digit = u32::cast_from((key >> shift) & digit_mask);
                    Atomic::fetch_add(&hist[digit as usize], 1u32);
                }
                col += units;
            }
            sync_cube();
            if unit == 0u32 {
                // The bin holding rank `remaining`: a counter loop that records the first bin whose
                // count exceeds what is left (no loop on a flag)
                let mut remaining = rank[0];
                // Runtime start values (a constant expression here would make the variable
                // comptime); `k` is below the keys counted, so a bin is always found
                let mut chosen = 0u32;
                let mut found = 0u32;
                let mut b = 0u32;
                while b < RADIX_BINS {
                    let count = Atomic::load(&hist[b as usize]);
                    if found == 0u32 {
                        if remaining < count {
                            chosen = b;
                            found = 1u32;
                        } else {
                            remaining -= count;
                        }
                    }
                    b += 1u32;
                }
                rank[0] = remaining;
                fixed[0] = prefix | (K::cast_from(chosen) << shift);
                fixed[1] = mask | (digit_mask << shift);
            }
            sync_cube();
        }
        if unit == 0u32 {
            out[row as usize] = F::reinterpret(fixed[0]);
        }
    }
}
