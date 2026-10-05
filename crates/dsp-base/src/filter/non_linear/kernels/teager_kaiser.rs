use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

use crate::core::read_extended;

/// Discrete Teager-Kaiser energy operator `ψ[t] = x[t]² − x[t−1] · x[t+1]` on a channel-major
/// `[channels, samples]` buffer. The neighbours of the end samples come from `edge` (an `EdgeMode`
/// id). Negative values (energy decreasing) are kept.
#[cube(launch)]
pub fn teager_kaiser_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    num_channels: u32,
    num_samples: u32,
    #[comptime] edge: u32,
) {
    let t = sample_position();
    let ch = channel_position();

    if t < num_samples && ch < num_channels {
        let base = (ch * num_samples) as usize;
        let curr = input[base + t as usize];
        let interior = t >= 1u32 && t + 1u32 < num_samples;
        // Extended positions with one sample of padding: t − 1 → t, t + 1 → t + 2
        let prev = if interior { input[base + (t - 1u32) as usize] } else { read_extended::<F>(input, base, num_samples, 1u32, t, edge) };
        let next = if interior { input[base + (t + 1u32) as usize] } else { read_extended::<F>(input, base, num_samples, 1u32, t + 2u32, edge) };
        output[base + t as usize] = curr * curr - prev * next;
    }
}
