use cubecl::prelude::*;

use crate::core::edge::EDGE_ODD;
use crate::core::read_extended_strided;

/// Values per section in the coefficient buffer: `[a1, a2, a3, m0, m1, m2]` (see [`crate::filter::Section::svf`]).
pub const SVF_COEFFS: usize = 6;

/// Floats in the coefficient buffer of an `n`-section cascade: `SVF_COEFFS · n` section coefficients,
/// the cascade DC gain, then `2n` at-rest states (the integrator states after a unit step has
/// settled, in the order the kernel keeps them).
pub const fn coeffs_len(n_sections: usize) -> usize {
    SVF_COEFFS * n_sections + 1 + 2 * n_sections
}

/// One pass of a cascade of `n_sections` second-order sections over a time block of one channel.
///
/// Sections run as trapezoidal state-variable filters (Cytomic SVF) whose coefficients are mapped
/// exactly from the designed SOS on the host in f64; unlike direct form II this stays accurate in
/// low precision when poles sit next to `z = 1`. The pass subtracts a per-channel offset (its first
/// sample, or the carried one) and adds `offset · dc_gain` back, so large DC levels do not eat the
/// mantissa; zero section state is then the steady state of that first sample.
///
/// Where a pass starts (block 0 without carried state):
/// - `rest == 0`: steady state of the first sample (scipy `sosfilt` with `zi · x[0]`, as
///   `sosfiltfilt` does).
/// - `rest != 0`: at rest, as if the input had been zero before (scipy `sosfilt` without `zi`): the
///   offset-domain input history is `−offset`, so the states start at `−offset ·` the at-rest states.
///
/// The logical sequence (the channel's `in_len` samples extended by `pad` odd-reflected samples on
/// each side, walked backwards when `reverse != 0`) is split into `num_blocks` blocks of
/// `block_len` steps. The cascade is linear, so blocks run in parallel (one unit per
/// `(channel, block)`, `ABSOLUTE_POS = block · num_channels + channel`):
///
/// - `phase = 0` (only when `num_blocks > 1`), blocks `0..num_blocks − 1`: run from zero state
///   (block 0 from the true initial state) and store the end state as the next block's start in
///   `block_states`; block 0 also stores the true initial state and offset in slot 0.
/// - between the phases [`sos_block_scan_kernel`] turns the stored zero-start end states into true
///   start states.
/// - `phase = 1`, all blocks: run from the block's start state and write logical samples
///   `[out_start, out_start + out_len)` to `output` (row length `out_len`); the last block writes
///   the final state and offset to `state`.
///
/// Memory order is given by strides: sample `t` of channel `c` is `input[c · in_channel_stride +
/// t · in_time_stride]` (channel-major: `(in_len, 1)`; time-major: `(1, channels)`), likewise for
/// `output`. Time-major buffers make the units of a plane (consecutive channels) read and write
/// consecutive addresses at every step.
///
/// - `coeffs`: [`coeffs_len`]`(n_sections)` values.
/// - `state`: `[channels][2·n_sections + 1]`, read when `carry != 0` (continuing a stream).
/// - `block_states`: `[channels][num_blocks][2·n_sections + 1]` start states (slot 0 holds the offset).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn sos_block_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    coeffs: &[F],
    state: &mut [F],
    block_states: &mut [F],
    num_channels: u32,
    in_len: u32,
    pad: u32,
    out_start: u32,
    out_len: u32,
    in_channel_stride: u32,
    in_time_stride: u32,
    out_channel_stride: u32,
    out_time_stride: u32,
    reverse: u32,
    carry: u32,
    rest: u32,
    block_len: u32,
    num_blocks: u32,
    #[comptime] phase: u32,
    #[comptime] n_sections: usize,
) {
    let active_blocks = if comptime!(phase == 0) { num_blocks - 1u32 } else { num_blocks };
    let unit = ABSOLUTE_POS as u32;
    if unit < num_channels * active_blocks {
        let b = unit / num_channels;
        let ch = unit - b * num_channels;
        let in_base = (ch * in_channel_stride) as usize;
        let out_base = (ch * out_channel_stride) as usize;
        let slot_len = n_sections * 2 + 1;
        let state_base = ch as usize * slot_len;
        let slots_base = (ch * num_blocks) as usize * slot_len;
        let total = in_len + 2u32 * pad;
        let dc_gain = coeffs[n_sections * SVF_COEFFS];
        let rest_base = n_sections * SVF_COEFFS + 1;
        let from_slot = comptime!(phase == 1) && num_blocks > 1u32;

        // Offset of this pass: carried, stored by phase 0, or the first logical sample
        let mut first = 0u32;
        if reverse != 0u32 {
            first = total - 1u32;
        }
        let offset = if from_slot {
            block_states[slots_base + n_sections * 2]
        } else if carry != 0u32 {
            state[state_base + n_sections * 2]
        } else {
            read_extended_strided::<F>(input, in_base, in_time_stride, in_len, pad, first, EDGE_ODD)
        };

        // Start state of this block
        let at_rest = !from_slot && carry == 0u32 && rest != 0u32 && b == 0u32;
        let mut z = Array::<F>::new(n_sections * 2);
        #[unroll]
        for k in 0..n_sections * 2 {
            let mut v = F::new(0.0f32);
            if from_slot {
                v = block_states[slots_base + b as usize * slot_len + k];
            } else if b == 0u32 && carry != 0u32 {
                v = state[state_base + k];
            } else if at_rest {
                v = -offset * coeffs[rest_base + k];
            }
            z[k] = v;
        }
        if comptime!(phase == 0) && b == 0u32 {
            #[unroll]
            for k in 0..n_sections * 2 {
                block_states[slots_base + k] = z[k];
            }
            block_states[slots_base + n_sections * 2] = offset;
        }
        let out_offset = offset * dc_gain;

        let i_end = u32::min((b + 1u32) * block_len, total);
        let mut i = b * block_len;
        while i < i_end {
            let mut j = i;
            if reverse != 0u32 {
                j = total - 1u32 - i;
            }
            let mut v = read_extended_strided::<F>(input, in_base, in_time_stride, in_len, pad, j, EDGE_ODD) - offset;

            #[unroll]
            for s in 0..n_sections {
                let c = s * SVF_COEFFS;
                let ic1 = z[2 * s];
                let ic2 = z[2 * s + 1];
                let v3 = v - ic2;
                let v1 = coeffs[c] * ic1 + coeffs[c + 1] * v3;
                let v2 = ic2 + coeffs[c + 1] * ic1 + coeffs[c + 2] * v3;
                z[2 * s] = F::new(2.0f32) * v1 - ic1;
                z[2 * s + 1] = F::new(2.0f32) * v2 - ic2;
                v = coeffs[c + 3] * v + coeffs[c + 4] * v1 + coeffs[c + 5] * v2;
            }

            if comptime!(phase == 1) {
                if j >= out_start && j - out_start < out_len {
                    output[out_base + ((j - out_start) * out_time_stride) as usize] = v + out_offset;
                }
            }
            i += 1u32;
        }

        if comptime!(phase == 0) {
            let next = slots_base + (b + 1u32) as usize * slot_len;
            #[unroll]
            for k in 0..n_sections * 2 {
                block_states[next + k] = z[k];
            }
        } else if b + 1u32 == num_blocks {
            #[unroll]
            for k in 0..n_sections * 2 {
                state[state_base + k] = z[k];
            }
            state[state_base + n_sections * 2] = offset;
        }
    }
}

/// Turns the zero-start end states left by phase 0 of [`sos_block_kernel`] into true block start
/// states, one unit per channel walking its blocks in order: `start[b] = Aᴸ · start[b − 1] +
/// start[b]` for `b ≥ 2` (slot 1 already holds block 0's true end state). `transition_delta` is
/// `Aᴸ − I` (row-major `[2·n_sections][2·n_sections]`) for the cascade's zero-input transition over
/// `block_len` steps: for slow poles `Aᴸ` is close to the identity, and applying
/// `prev + (s + (Aᴸ − I)·prev)` keeps rounding on the small part only.
#[cube(launch)]
pub fn sos_block_scan_kernel<F: Float>(
    block_states: &mut [F],
    transition_delta: &[F],
    num_channels: u32,
    num_blocks: u32,
    #[comptime] n_sections: usize,
) {
    let ch = ABSOLUTE_POS_X;
    if ch < num_channels {
        let dim = n_sections * 2;
        let slot_len = dim + 1;
        let slots_base = (ch * num_blocks) as usize * slot_len;
        let mut prev = Array::<F>::new(n_sections * 2);
        #[unroll]
        for k in 0..n_sections * 2 {
            prev[k] = block_states[slots_base + slot_len + k];
        }
        let mut b = 2u32;
        while b < num_blocks {
            let at = slots_base + b as usize * slot_len;
            let mut next = Array::<F>::new(n_sections * 2);
            #[unroll]
            for r in 0..n_sections * 2 {
                let mut acc = block_states[at + r];
                #[unroll]
                for c in 0..n_sections * 2 {
                    acc += transition_delta[r * dim + c] * prev[c];
                }
                next[r] = prev[r] + acc;
            }
            #[unroll]
            for k in 0..n_sections * 2 {
                block_states[at + k] = next[k];
                prev[k] = next[k];
            }
            b += 1u32;
        }
    }
}
