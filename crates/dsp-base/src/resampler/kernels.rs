use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

use crate::core::read_extended;
use crate::core::edge::EDGE_ZEROS;

/// `output[c, m] = input[c, phase + m · step]` (keep every `step`-th sample).
#[cube(launch)]
pub fn downsample_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
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
    input: &Array<F>,
    taps: &Array<F>,
    output: &mut Array<F>,
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
        let mut acc = F::new(0.0f32);
        let mut j = phase;
        while j < i32::cast_from(num_taps) {
            let s = (i0 - j) / up_i;
            if s >= 0i32 && s < i32::cast_from(in_len) {
                acc += taps[u32::cast_from(j) as usize] * input[base + u32::cast_from(s) as usize];
            } else if comptime!(edge != EDGE_ZEROS) {
                let ext = u32::cast_from(s + i32::cast_from(ext_pad));
                acc += taps[u32::cast_from(j) as usize] * read_extended::<F>(input, base, in_len, ext_pad, ext, edge);
            }
            j += up_i;
        }
        output[(ch * out_len + m) as usize] = acc;
    }
}
