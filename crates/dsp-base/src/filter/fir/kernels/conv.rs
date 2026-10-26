use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

use crate::core::read_extended;

/// Causal FIR `y[t] = Σ_k taps[k] · x[t − k]` on a channel-major `[channels, samples]` buffer, one unit
/// per `(channel, sample)`. Samples before the row start come from `edge` (an `EdgeMode` id; scipy
/// `lfilter` reads zeros). Units with a full history take a branch-free path.
#[cube(launch)]
pub fn fir_filter_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    taps: &[F],
    num_channels: u32,
    num_samples: u32,
    num_taps: u32,
    #[comptime] edge: u32,
) {
    let t = sample_position();
    let ch = channel_position();

    if t < num_samples && ch < num_channels {
        let base = (ch * num_samples) as usize;
        let history = num_taps - 1u32;
        let mut sum = F::new(0.0f32);
        let mut k = 0u32;
        if t >= history {
            while k < num_taps {
                sum += input[base + (t - k) as usize] * taps[k as usize];
                k += 1u32;
            }
        } else {
            // Logical sample t − k is extended position t − k + history
            while k < num_taps {
                sum += read_extended::<F>(input, base, num_samples, history, t + history - k, edge) * taps[k as usize];
                k += 1u32;
            }
        }
        output[base + t as usize] = sum;
    }
}

/// Centered FIR `y[t] = Σ_k taps[k] · x[t + k − radius]` (`2·radius + 1` taps, zero phase for symmetric
/// taps). Samples beyond either end come from `edge` (an `EdgeMode` id). Units whose window lies
/// inside the row take a branch-free path.
#[cube(launch)]
pub fn fir_centered_filter_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    taps: &[F],
    num_channels: u32,
    num_samples: u32,
    radius: u32,
    #[comptime] edge: u32,
) {
    let t = sample_position();
    let ch = channel_position();

    if t < num_samples && ch < num_channels {
        let base = (ch * num_samples) as usize;
        let num_taps = 2u32 * radius + 1u32;
        let mut sum = F::new(0.0f32);
        let mut k = 0u32;
        if t >= radius && t + radius < num_samples {
            let first = base + (t - radius) as usize;
            while k < num_taps {
                sum += input[first + k as usize] * taps[k as usize];
                k += 1u32;
            }
        } else {
            // Logical sample t + k − radius is extended position t + k
            while k < num_taps {
                sum += read_extended::<F>(input, base, num_samples, radius, t + k, edge) * taps[k as usize];
                k += 1u32;
            }
        }
        output[base + t as usize] = sum;
    }
}

/// Fills `tile` (`tile_y` rows of `span` values) with extended positions `t0 + k` (`k < span`,
/// padding `pad`) of the cube's channels; rows past the last channel hold zeros.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_tile<F: Float>(
    input: &[F],
    tile: &mut Shared<[F]>,
    num_channels: u32,
    num_samples: u32,
    pad: u32,
    #[comptime] span: u32,
    #[comptime] tile_x: u32,
    #[comptime] edge: u32,
) {
    let ch = channel_position();
    let t0 = sample_position() - UNIT_POS_X;
    let row = UNIT_POS_Y * span;
    let mut k = UNIT_POS_X;
    while k < span {
        let mut v = F::new(0.0f32);
        if ch < num_channels {
            v = read_extended::<F>(input, (ch * num_samples) as usize, num_samples, pad, t0 + k, edge);
        }
        tile[(row + k) as usize] = v;
        k += tile_x;
    }
}

/// [`fn@fir_filter_kernel`] through shared memory: each cube loads its `tile_x` samples of `tile_y`
/// channels plus the `num_taps − 1` samples of history once, then every unit reads its window from
/// the tile. `num_taps`, `tile_x` (= `CUBE_DIM_X`) and `tile_y` (= `CUBE_DIM_Y`) are comptime.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn fir_tiled_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    taps: &[F],
    num_channels: u32,
    num_samples: u32,
    #[comptime] num_taps: u32,
    #[comptime] tile_x: u32,
    #[comptime] tile_y: u32,
    #[comptime] edge: u32,
) {
    let span = comptime!(tile_x + num_taps - 1);
    let mut tile = Shared::<[F]>::new_slice(comptime!((span * tile_y) as usize));
    load_tile::<F>(input, &mut tile, num_channels, num_samples, comptime!(num_taps - 1), span, tile_x, edge);
    sync_cube();

    let t = sample_position();
    let ch = channel_position();
    if t < num_samples && ch < num_channels {
        // y[t] = Σ_k taps[k] · x[t − k]; x[t − k] sits at tile column UNIT_POS_X + (num_taps − 1) − k
        let at = UNIT_POS_Y * span + UNIT_POS_X + comptime!(num_taps - 1);
        let mut sum = F::new(0.0f32);
        let mut k = 0u32;
        while k < num_taps {
            sum += tile[(at - k) as usize] * taps[k as usize];
            k += 1u32;
        }
        output[(ch * num_samples + t) as usize] = sum;
    }
}

/// [`fn@fir_centered_filter_kernel`] through shared memory: each cube loads its `tile_x` samples of
/// `tile_y` channels plus `radius` samples on each side once. `radius`, `tile_x` (= `CUBE_DIM_X`) and
/// `tile_y` (= `CUBE_DIM_Y`) are comptime.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn fir_centered_tiled_kernel<F: Float>(
    input: &[F],
    output: &mut [F],
    taps: &[F],
    num_channels: u32,
    num_samples: u32,
    #[comptime] radius: u32,
    #[comptime] tile_x: u32,
    #[comptime] tile_y: u32,
    #[comptime] edge: u32,
) {
    let span = comptime!(tile_x + 2 * radius);
    let mut tile = Shared::<[F]>::new_slice(comptime!((span * tile_y) as usize));
    load_tile::<F>(input, &mut tile, num_channels, num_samples, radius, span, tile_x, edge);
    sync_cube();

    let t = sample_position();
    let ch = channel_position();
    if t < num_samples && ch < num_channels {
        // y[t] = Σ_k taps[k] · x[t + k − radius]; x[t + k − radius] sits at tile column UNIT_POS_X + k
        let at = UNIT_POS_Y * span + UNIT_POS_X;
        let mut sum = F::new(0.0f32);
        let mut k = 0u32;
        while k < comptime!(2 * radius + 1) {
            sum += tile[(at + k) as usize] * taps[k as usize];
            k += 1u32;
        }
        output[(ch * num_samples + t) as usize] = sum;
    }
}
