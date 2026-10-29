use cubecl::prelude::*;
use dspk_core::compute::{row_position, row_share};

/// Bits of one radix digit of [`fn@row_kth_radix_kernel`].
pub const RADIX_BITS: u32 = 8;

/// Bins of a digit's histogram.
pub const RADIX_BINS: u32 = 1 << RADIX_BITS;

/// What [`fn@row_kth_radix_kernel`] ranks: `|x|`.
pub const KEY_ABS: u32 = 0;
/// What [`fn@row_kth_radix_kernel`] ranks: `x` itself (signed).
pub const KEY_SIGNED: u32 = 1;
/// What [`fn@row_kth_radix_kernel`] ranks: `|x − center[row]|`.
pub const KEY_ABS_DEV: u32 = 2;

/// `x`'s bits as an unsigned key `K` that sorts like the value: positive floats get the sign bit
/// set, negative ones all bits flipped (more negative → smaller key).
#[cube]
fn order_key<F: Float, K: Int>(x: F, #[comptime] key_bits: u32) -> K {
    let bits = K::reinterpret(x);
    let top = K::cast_from(1u32) << K::cast_from(comptime!(key_bits - 1));
    let flip = if (bits & top) == top { !K::cast_from(0u32) } else { top };
    bits ^ flip
}

/// The float of an [`fn@order_key`] key.
#[cube]
fn key_value<F: Float, K: Int>(key: K, #[comptime] key_bits: u32) -> F {
    let top = K::cast_from(1u32) << K::cast_from(comptime!(key_bits - 1));
    let flip = if (key & top) == top { top } else { !K::cast_from(0u32) };
    F::reinterpret(key ^ flip)
}

/// The sort key of `x` for `mode` ([`KEY_ABS`], [`KEY_SIGNED`], [`KEY_ABS_DEV`] around `center`).
#[cube]
fn select_key<F: Float, K: Int>(x: F, center: F, #[comptime] key_bits: u32, #[comptime] mode: u32) -> K {
    if comptime!(mode == KEY_SIGNED) {
        order_key::<F, K>(x, key_bits)
    } else if comptime!(mode == KEY_ABS_DEV) {
        K::reinterpret(F::abs(x - center))
    } else {
        K::reinterpret(F::abs(x))
    }
}

/// One cube per row: the `k`-th smallest (0-based) key among columns `col_start..col_start + cols`
/// of row `row` (rows `row_stride` apart), exactly, written as the value it came from. The key is
/// set by `mode` (comptime): `|x|` ([`KEY_ABS`]), `x` ([`KEY_SIGNED`]) or `|x − center[row]|`
/// ([`KEY_ABS_DEV`]; `center` holds one value per row, other modes ignore it).
///
/// The bits of a non-negative IEEE float, read as an unsigned integer `K` of the same width, sort
/// like its value (subnormals and zero included; NaN above +∞); signed values sort the same after
/// [`fn@order_key`] (sign bit set for positives, all bits flipped for negatives). So the `k`-th
/// smallest value is the `k`-th smallest key, found digit by digit from the most significant ([`RADIX_BITS`] each): a
/// pass counts, in a shared-memory histogram, the keys that agree with the digits fixed so far,
/// and one unit walks the histogram to the bin holding rank `k`, fixes that digit and subtracts the
/// keys in lower bins from `k`. After `key_bits / RADIX_BITS` passes over the row the key is the
/// exact bit pattern of the answer. `key_bits` is the width of `F` and `K` (from the host:
/// `K::size_bits()` inside a kernel does not give it); `units` is the cube's x size.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn row_kth_radix_kernel<F: Float, K: Int>(
    input: &[F],
    center: &[F],
    out: &mut [F],
    rows: u32,
    row_stride: u32,
    col_start: u32,
    cols: u32,
    k: u32,
    #[comptime] key_bits: u32,
    #[comptime] units: u32,
    #[comptime] mode: u32,
) {
    let row = row_position();
    // `row` is uniform across the cube, so every unit of a cube takes the same branch
    if row < rows {
        let unit = UNIT_POS_X;
        let base = row * row_stride + col_start;
        // Only the deviation mode reads the centre (comptime: the others never load it)
        let c = if comptime!(mode == KEY_ABS_DEV) { center[row as usize] } else { F::new(0.0f32) };
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
            let share = row_share(cols, units);
            let mut col = share.start;
            while col < share.end {
                let key = select_key::<F, K>(input[(base + col) as usize], c, key_bits, mode);
                if (key & mask) == prefix {
                    let digit = u32::cast_from((key >> shift) & digit_mask);
                    Atomic::fetch_add(&hist[digit as usize], 1u32);
                }
                col += share.step;
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
            // The value of the key found: signed keys are encoded, |x| keys are the bits themselves
            out[row as usize] = if comptime!(mode == KEY_SIGNED) { key_value::<F, K>(fixed[0], key_bits) } else { F::reinterpret(fixed[0]) };
        }
    }
}
