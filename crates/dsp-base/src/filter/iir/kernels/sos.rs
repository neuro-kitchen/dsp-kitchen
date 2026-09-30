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

/// Cascade of `n_sections` second-order sections, one thread per channel, all sections applied per
/// sample in registers.
///
/// Sections run as trapezoidal state-variable filters (Cytomic SVF) whose coefficients are mapped
/// exactly from the designed SOS on the host in f64. Unlike direct form II, this stays accurate in
/// f32 when poles sit next to `z = 1` (low cutoffs, narrow notches). The pass also subtracts a
/// per-channel offset (its first sample) and adds `offset · dc_gain` back, so large DC levels do not
/// eat the f32 mantissa; starting from zero section state is then the steady state (no step).
///
/// - `coeffs`: `SVF_COEFFS` floats per section followed by the cascade DC gain.
/// - `state`: `[channels][2·n_sections + 1]` = section integrator states then the offset; read when
///   `carry != 0` (continuing a stream), always written with the final state.
/// - The input channel (`in_len` samples) is read as if extended by `pad` odd-reflected samples on
///   each side; logical samples `[out_start, out_start + out_len)` are written to `output`
///   (row length `out_len`).
/// - `reverse != 0` processes the logical sequence from last to first (backward pass without a copy).
#[cube(launch)]
pub fn sos_cascade_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    coeffs: &Array<f32>,
    state: &mut Array<f32>,
    num_channels: u32,
    in_len: u32,
    pad: u32,
    out_start: u32,
    out_len: u32,
    reverse: u32,
    carry: u32,
    #[comptime] n_sections: usize,
) {
    let ch = ABSOLUTE_POS_X;
    if ch < num_channels {
        let in_base = (ch * in_len) as usize;
        let out_base = (ch * out_len) as usize;
        let state_len = n_sections * 2 + 1;
        let state_base = ch as usize * state_len;
        let total = in_len + 2u32 * pad;
        let dc_gain = coeffs[n_sections * SVF_COEFFS];

        let mut z = Array::<f32>::new(n_sections * 2);
        #[unroll]
        for k in 0..n_sections * 2 {
            let mut v = 0.0f32;
            if carry != 0u32 {
                v = state[state_base + k];
            }
            z[k] = v;
        }
        let mut first = 0u32;
        if reverse != 0u32 {
            first = total - 1u32;
        }
        let offset = if carry != 0u32 {
            state[state_base + n_sections * 2]
        } else {
            read_odd_extended(input, in_base, in_len, pad, first)
        };
        let out_offset = offset * dc_gain;

        let mut i = 0u32;
        while i < total {
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

            if j >= out_start && j - out_start < out_len {
                output[out_base + (j - out_start) as usize] = v + out_offset;
            }
            i += 1u32;
        }

        #[unroll]
        for k in 0..n_sections * 2 {
            state[state_base + k] = z[k];
        }
        state[state_base + n_sections * 2] = offset;
    }
}
