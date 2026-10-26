use cubecl::prelude::*;

/// How a stored value's bits are read (`kind` of [`fn@unpack_stored_kernel`]).
pub const SIGNED: u32 = 0;
pub const UNSIGNED: u32 = 1;
pub const FLOAT: u32 = 2;

/// Converts `total` stored values (`bytes` each, packed little-endian into `words`, channel-major
/// rows of `num_samples`) to `output[i] = value · gains[ch] + offsets[ch]`. One unit per value
/// (`ABSOLUTE_POS`).
#[cube(launch)]
pub fn unpack_stored_kernel<F: Float>(
    words: &[u32],
    gains: &[F],
    offsets: &[F],
    output: &mut [F],
    num_samples: u32,
    total: u32,
    #[comptime] bytes: u32,
    #[comptime] kind: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total {
        let ch = idx / num_samples;
        let per_word = comptime!(4 / bytes);
        let bits = comptime!(bytes * 8);
        let raw = if comptime!(bytes == 4) {
            words[idx as usize]
        } else {
            let word = words[(idx / per_word) as usize];
            let shift = (idx % per_word) * bits;
            (word >> shift) & comptime!((1u32 << (bytes * 8)) - 1)
        };
        let value = if comptime!(kind == FLOAT) {
            F::cast_from(f32::reinterpret(raw))
        } else if comptime!(kind == UNSIGNED) {
            F::cast_from(raw)
        } else if comptime!(bytes == 4) {
            // Two's complement: negate the magnitude so small negative values stay exact
            if raw >= 0x8000_0000u32 { -F::cast_from((!raw) + 1u32) } else { F::cast_from(raw) }
        } else {
            let half = comptime!(1u32 << (bytes * 8 - 1));
            if raw >= half { F::cast_from(raw) - F::cast_from(comptime!(1u32 << (bytes * 8))) } else { F::cast_from(raw) }
        };
        output[idx as usize] = value * gains[ch as usize] + offsets[ch as usize];
    }
}
