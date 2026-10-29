use cubecl::prelude::*;
use dspk_core::compute::{channel_position, element_position, sample_position};

use crate::core::read_extended;
use crate::core::edge::EDGE_ZEROS;

/// `output[c, m] = input[c, phase + m · step]` (keep every `step`-th sample).
#[cube(launch)]
pub fn downsample_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    num_channels: u32,
    in_len: u32,
    out_len: u32,
    step: u32,
    phase: u32,
) {
    let m = sample_position();
    let ch = channel_position();
    if m < out_len && ch < num_channels {
        output[(ch * out_len + m) as usize] = input[(ch * in_len + phase + m * step) as usize];
    }
}

/// Polyphase FIR resampling (`upfirdn` restricted to the outputs `resample_poly` keeps): output `m`
/// is `Σ_j taps[j] · u[i₀ − j]` with `i₀ = m · down + start`, where `u` is the input up-sampled by
/// `up` (`u[i] = x[i / up]` when `up` divides `i`, else 0). Only taps aligned with input samples are
/// visited (`j ≡ i₀ mod up`). Input positions outside the row come from `edge` (an `EdgeMode` id;
/// `resample_poly` pads with zeros); `ext_pad` is how far outside they can reach.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn upfirdn_kernel<F: Float>(
    input: &[F],
    taps: &[F],
    output: &mut [F],
    num_channels: u32,
    in_len: u32,
    out_len: u32,
    num_taps: u32,
    up: u32,
    down: u32,
    start: i32,
    ext_pad: u32,
    #[comptime] edge: u32,
) {
    let m = sample_position();
    let ch = channel_position();
    if m < out_len && ch < num_channels {
        let base = (ch * in_len) as usize;
        let up_i = i32::cast_from(up);
        let i0 = i32::cast_from(m) * i32::cast_from(down) + start;
        let phase = ((i0 % up_i) + up_i) % up_i;
        let n_taps = i32::cast_from(num_taps);
        let len = i32::cast_from(in_len);
        let mut acc = F::new(0.0f32);
        // Tap `j = phase + k · up` reads input `s = (i0 − j) / up = s0 − k`: one exact division
        // (`i0 − phase` is a multiple of `up`), then a decrement per tap
        let s0 = (i0 - phase) / up_i;
        let visits = if phase < n_taps { (n_taps - phase + up_i - 1) / up_i } else { 0i32 };
        let mut j = phase;
        let mut s = s0;
        if s0 < len && s0 - visits + 1 >= 0i32 {
            // The whole window lies inside the row (nearly every output): no bounds checks
            let mut k = 0i32;
            while k < visits {
                acc += taps[u32::cast_from(j) as usize] * input[base + u32::cast_from(s) as usize];
                j += up_i;
                s -= 1i32;
                k += 1i32;
            }
        } else {
            while j < n_taps {
                if s >= 0i32 && s < len {
                    acc += taps[u32::cast_from(j) as usize] * input[base + u32::cast_from(s) as usize];
                } else if comptime!(edge != EDGE_ZEROS) {
                    let ext = u32::cast_from(s + i32::cast_from(ext_pad));
                    acc += taps[u32::cast_from(j) as usize] * read_extended::<F>(input, base, in_len, ext_pad, ext, edge);
                }
                j += up_i;
                s -= 1i32;
            }
        }
        output[(ch * out_len + m) as usize] = acc;
    }
}

/// `⌊a / b⌋` for `b > 0` (integer division truncates toward zero).
#[cube]
fn floor_div(a: i32, b: i32) -> i32 {
    let q = a / b;
    if a % b != 0i32 && a < 0i32 { q - 1i32 } else { q }
}

/// [`fn@upfirdn_kernel`] with `N` consecutive outputs per unit (`out_len` a multiple of `N`).
/// Output `m + l` meets input sample `s` at tap `(m + l) · down + start − s · up`; `table[d +
/// reach]` holds in lane `l` the tap `d + l · down` (zero outside the filter), `reach = (N − 1) ·
/// down`, so every load is aligned. Input samples are visited newest first, so each lane adds its
/// taps in the scalar kernel's order (plus exact zeros): outputs are bit-identical to it.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn upfirdn_vector_kernel<F: Float, N: Size>(
    input: &[F],
    table: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
    in_len: u32,
    out_len: u32,
    num_taps: u32,
    up: u32,
    down: u32,
    start: i32,
    ext_pad: u32,
    #[comptime] edge: u32,
) {
    let v = element_position();
    if v < output.len() {
        let n = N::value() as u32;
        let per_row = out_len / n;
        let m = (v as u32 % per_row) * n;
        let base = ((v as u32 / per_row) * in_len) as usize;
        let up_i = i32::cast_from(up);
        let down_i = i32::cast_from(down);
        let len = i32::cast_from(in_len);
        let i0 = i32::cast_from(m) * down_i + start;
        let reach = i32::cast_from(n - 1u32) * down_i;
        // Inputs some lane meets: s · up in [i0 − (K − 1), i0 + reach]
        let s_hi = floor_div(i0 + reach, up_i);
        let s_lo = 0i32 - floor_div(i32::cast_from(num_taps) - 1i32 - i0, up_i);
        let mut acc = Vector::<F, N>::new(F::new(0.0f32));
        let mut s = s_hi;
        if s_lo >= 0i32 && s_hi < len {
            while s >= s_lo {
                let d = i0 - s * up_i + reach;
                acc += table[u32::cast_from(d) as usize] * Vector::new(input[base + u32::cast_from(s) as usize]);
                s -= 1i32;
            }
        } else {
            while s >= s_lo {
                let d = i0 - s * up_i + reach;
                if s >= 0i32 && s < len {
                    acc += table[u32::cast_from(d) as usize] * Vector::new(input[base + u32::cast_from(s) as usize]);
                } else if comptime!(edge != EDGE_ZEROS) {
                    let ext = u32::cast_from(s + i32::cast_from(ext_pad));
                    acc += table[u32::cast_from(d) as usize] * Vector::new(read_extended::<F>(input, base, in_len, ext_pad, ext, edge));
                }
                s -= 1i32;
            }
        }
        output[v] = acc;
    }
}
