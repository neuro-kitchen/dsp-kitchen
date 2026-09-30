use cubecl::prelude::*;

/// Floats per section in the coefficient buffer: `[a1, a2, a3, m0, m1, m2]` (see [`crate::filter::Section::svf`]).
pub const SVF_COEFFS: usize = 6;

/// Reads logical sample `j` of a channel extended by `pad` samples of odd reflection on each side
/// (`2·x[0] − x[k]` before the start, `2·x[n−1] − x[n−1−k]` after the end).
#[cube]
fn read_odd_extended(input: &Array<f32>, base: usize, len: u32, pad: u32, j: u32) -> f32 {
    if j < pad {
        2.0f32 * input[base] - input[base + (pad - j) as usize]
    } else if j - pad < len {
        input[base + (j - pad) as usize]
    } else {
        let k = j - pad - len + 1u32;
        let last = base + (len - 1u32) as usize;
        2.0f32 * input[last] - input[last - k as usize]
    }
}

/// One pass of a cascade of `n_sections` second-order sections over a time block of one channel.
///
/// Sections run as trapezoidal state-variable filters (Cytomic SVF) whose coefficients are mapped
/// exactly from the designed SOS on the host in f64; unlike direct form II this stays accurate in
/// f32 when poles sit next to `z = 1`. The pass subtracts a per-channel offset (its first sample,
/// or the carried one) and adds `offset · dc_gain` back, so large DC levels do not eat the f32
/// mantissa; zero section state is then the steady state.
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
/// - `coeffs`: `SVF_COEFFS` floats per section followed by the cascade DC gain.
/// - `state`: `[channels][2·n_sections + 1]`, read when `carry != 0` (continuing a stream).
/// - `block_states`: `[channels][num_blocks][2·n_sections + 1]` start states (slot 0 holds the offset).
#[cube(launch)]
pub fn sos_block_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    coeffs: &Array<f32>,
    state: &mut Array<f32>,
    block_states: &mut Array<f32>,
    num_channels: u32,
    in_len: u32,
    pad: u32,
    out_start: u32,
    out_len: u32,
    reverse: u32,
    carry: u32,
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
        let in_base = (ch * in_len) as usize;
        let out_base = (ch * out_len) as usize;
        let slot_len = n_sections * 2 + 1;
        let state_base = ch as usize * slot_len;
        let slots_base = (ch * num_blocks) as usize * slot_len;
        let total = in_len + 2u32 * pad;
        let dc_gain = coeffs[n_sections * SVF_COEFFS];

        // Start state and offset of this block
        let mut z = Array::<f32>::new(n_sections * 2);
        let from_slot = comptime!(phase == 1) && num_blocks > 1u32;
        #[unroll]
        for k in 0..n_sections * 2 {
            let mut v = 0.0f32;
            if from_slot {
                v = block_states[slots_base + b as usize * slot_len + k];
            } else if b == 0u32 && carry != 0u32 {
                v = state[state_base + k];
            }
            z[k] = v;
        }
        let mut first = 0u32;
        if reverse != 0u32 {
            first = total - 1u32;
        }
        let offset = if from_slot {
            block_states[slots_base + n_sections * 2]
        } else if carry != 0u32 {
            state[state_base + n_sections * 2]
        } else {
            read_odd_extended(input, in_base, in_len, pad, first)
        };
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
            let mut v = read_odd_extended(input, in_base, in_len, pad, j) - offset;

            #[unroll]
            for s in 0..n_sections {
                let c = s * SVF_COEFFS;
                let ic1 = z[2 * s];
                let ic2 = z[2 * s + 1];
                let v3 = v - ic2;
                let v1 = coeffs[c] * ic1 + coeffs[c + 1] * v3;
                let v2 = ic2 + coeffs[c + 1] * ic1 + coeffs[c + 2] * v3;
                z[2 * s] = 2.0f32 * v1 - ic1;
                z[2 * s + 1] = 2.0f32 * v2 - ic2;
                v = coeffs[c + 3] * v + coeffs[c + 4] * v1 + coeffs[c + 5] * v2;
            }

            if comptime!(phase == 1) {
                if j >= out_start && j - out_start < out_len {
                    output[out_base + (j - out_start) as usize] = v + out_offset;
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
/// `prev + (s + (Aᴸ − I)·prev)` keeps f32 rounding on the small part only.
#[cube(launch)]
pub fn sos_block_scan_kernel(
    block_states: &mut Array<f32>,
    transition_delta: &Array<f32>,
    num_channels: u32,
    num_blocks: u32,
    #[comptime] n_sections: usize,
) {
    let ch = ABSOLUTE_POS_X;
    if ch < num_channels {
        let dim = n_sections * 2;
        let slot_len = dim + 1;
        let slots_base = (ch * num_blocks) as usize * slot_len;
        let mut prev = Array::<f32>::new(n_sections * 2);
        #[unroll]
        for k in 0..n_sections * 2 {
            prev[k] = block_states[slots_base + slot_len + k];
        }
        let mut b = 2u32;
        while b < num_blocks {
            let at = slots_base + b as usize * slot_len;
            let mut next = Array::<f32>::new(n_sections * 2);
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
