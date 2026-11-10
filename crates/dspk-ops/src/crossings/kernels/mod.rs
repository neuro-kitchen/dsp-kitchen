//! Level-crossing kernels for [`super`]: count the crossings of each `(block, channel)`, write them
//! compacted, the extremum of each stretch between crossings, and one column of a buffer.
//!
//! Values are compared in *signed* space, `s · x` with `s = +1` (rising: a level above the
//! baseline) or `s = −1` (falling: a level below it), so one kernel serves both directions:
//! - an **up** crossing at `t`: `s · x[t−1] < level ≤ s · x[t]`;
//! - a **down** crossing at `t`: `s · x[t−1] ≥ release > s · x[t]`.
//!
//! At `t = 0` the previous value is `prev[ch]` (the last sample of the window before, or NaN for
//! none: NaN compares false, so the first sample of a recording never crosses).

use cubecl::prelude::*;
use dspk_core::compute::{channel_position, sample_position};

/// Kind ids written by [`write_crossings_kernel`].
pub const CROSSING_UP: u32 = 0;
pub const CROSSING_DOWN: u32 = 1;

/// Crossing kind at local sample `t` of the row starting at `row`: [`CROSSING_UP`],
/// [`CROSSING_DOWN`], or 2 for none.
#[cube]
fn crossing<F: Float>(trace: &[F], prev: &[F], ch: u32, row: u32, t: u32, sign: F, level: F, release: F) -> u32 {
    let mut p = prev[ch as usize];
    if t > 0u32 {
        p = trace[(row + t - 1u32) as usize];
    }
    let sp = sign * p;
    let sv = sign * trace[(row + t) as usize];
    let mut kind = 2u32;
    if sp < level && sv >= level {
        kind = CROSSING_UP;
    } else if sp >= release && sv < release {
        kind = CROSSING_DOWN;
    }
    kind
}

/// Counts the crossings of each `(block, channel)` unit (contiguous blocks of `block` samples from
/// `scan_start`) into `block_counts[ch · num_blocks + b]`.
#[cube(launch)]
pub fn count_crossings_kernel<F: Float + CubeElement + LaunchArg>(
    trace: &[F],
    prev: &[F],
    levels: &[F],
    releases: &[F],
    block_counts: &mut [u32],
    num_channels: u32,
    num_samples: u32,
    scan_start: u32,
    scan_end: u32,
    num_blocks: u32,
    block: u32,
    sign: F,
) {
    let b = sample_position();
    let ch = channel_position();
    if ch < num_channels && b < num_blocks {
        let (level, release) = (levels[ch as usize], releases[ch as usize]);
        let row = ch * num_samples;
        let mut n = 0u32;
        let mut t = scan_start + b * block;
        let mut end = t + block;
        if end > scan_end {
            end = scan_end;
        }
        while t < end {
            if crossing::<F>(trace, prev, ch, row, t, sign, level, release) < 2u32 {
                n += 1u32;
            }
            t += 1u32;
        }
        block_counts[(ch * num_blocks + b) as usize] = n;
    }
}

/// Writes each unit's crossings (local sample and kind) from `channel_bases[ch] +
/// block_offsets[ch · num_blocks + b]` on: every channel's crossings come out contiguous and in
/// time order (blocks are contiguous and in order).
#[cube(launch)]
pub fn write_crossings_kernel<F: Float + CubeElement + LaunchArg>(
    trace: &[F],
    prev: &[F],
    levels: &[F],
    releases: &[F],
    block_offsets: &[u32],
    channel_bases: &[u32],
    out_indices: &mut [u32],
    out_kinds: &mut [u32],
    num_channels: u32,
    num_samples: u32,
    scan_start: u32,
    scan_end: u32,
    num_blocks: u32,
    block: u32,
    sign: F,
) {
    let b = sample_position();
    let ch = channel_position();
    if ch < num_channels && b < num_blocks {
        let (level, release) = (levels[ch as usize], releases[ch as usize]);
        let row = ch * num_samples;
        let mut slot = channel_bases[ch as usize] + block_offsets[(ch * num_blocks + b) as usize];
        let mut t = scan_start + b * block;
        let mut end = t + block;
        if end > scan_end {
            end = scan_end;
        }
        while t < end {
            let kind = crossing::<F>(trace, prev, ch, row, t, sign, level, release);
            if kind < 2u32 {
                out_indices[slot as usize] = t;
                out_kinds[slot as usize] = kind;
                slot += 1u32;
            }
            t += 1u32;
        }
    }
}

/// One unit per stretch `i`: the largest signed value `s · x` over local samples
/// `starts[i]..ends[i]` of channel `channels[i]`, and where it is (the first such sample). An empty
/// stretch gives `−3e38` at its start.
#[cube(launch)]
pub fn stretch_extrema_kernel<F: Float + CubeElement + LaunchArg>(
    trace: &[F],
    channels: &[u32],
    starts: &[u32],
    ends: &[u32],
    out_values: &mut [F],
    out_indices: &mut [u32],
    num_stretches: u32,
    num_samples: u32,
    sign: F,
) {
    let i = ABSOLUTE_POS_X;
    if i < num_stretches {
        let row = channels[i as usize] * num_samples;
        let start = starts[i as usize];
        let mut best = F::new(-3.0e38f32);
        let mut at = start;
        let mut t = start;
        while t < ends[i as usize] {
            let v = sign * trace[(row + t) as usize];
            if v > best {
                best = v;
                at = t;
            }
            t += 1u32;
        }
        out_values[i as usize] = best;
        out_indices[i as usize] = at;
    }
}

/// `out[ch] = trace[ch · num_samples + col]`: one column of a `[channels, samples]` buffer.
#[cube(launch)]
pub fn column_kernel<F: Float>(trace: &[F], out: &mut [F], num_channels: u32, num_samples: u32, col: u32) {
    let ch = ABSOLUTE_POS_X;
    if ch < num_channels {
        out[ch as usize] = trace[(ch * num_samples + col) as usize];
    }
}
