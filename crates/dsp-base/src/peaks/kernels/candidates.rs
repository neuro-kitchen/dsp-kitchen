//! Candidate-peak compaction kernels: count per `(block, channel)`, scan the counts into offsets,
//! write the candidates. A candidate is a local extremum (`x[t−1] < x[t] ≥ x[t+1]` for maxima,
//! mirrored for minima) whose signed value reaches its channel's height.

use cubecl::prelude::*;
use dsp_core::compute::{channel_position, sample_position};

/// Polarity ids of the kernels (see [`Polarity`](crate::peaks::Polarity)).
pub const POLARITY_POSITIVE: u32 = 0;
pub const POLARITY_NEGATIVE: u32 = 1;
pub const POLARITY_BOTH: u32 = 2;

/// First sample scanned by unit `b`; it then steps by `lanes`. Units are grouped `lanes` at a
/// time (the runtime's plane size) and a group's lanes interleave over its `lanes · block` samples,
/// so the units of a plane read neighbouring addresses. With one-unit planes every unit scans
/// `block` contiguous samples.
#[cube]
fn first_candidate_sample(scan_start: u32, b: u32, lanes: u32, block: u32) -> u32 {
    let group = b / lanes;
    scan_start + group * lanes * block + (b - group * lanes)
}

/// Whether local sample `t` of the row starting at `row` is a candidate (`1 ≤ t < samples − 1`).
#[cube]
fn is_candidate<F: Float>(trace: &[F], row: u32, t: u32, height: F, polarity: u32) -> bool {
    let v = trace[(row + t) as usize];
    let prev = trace[(row + t - 1u32) as usize];
    let next = trace[(row + t + 1u32) as usize];
    let neg_height = F::new(0.0f32) - height;
    let is_max = v >= height && prev < v && v >= next;
    let is_min = v <= neg_height && prev > v && v <= next;
    (polarity != POLARITY_NEGATIVE && is_max) || (polarity != POLARITY_POSITIVE && is_min)
}

/// Counts the candidates of each `(block, channel)` unit (block = [`fn@sample_position`], channel =
/// [`fn@channel_position`]) into `block_counts[ch · num_blocks + b]`.
#[cube(launch)]
pub fn count_peak_candidates_kernel<F: Float>(
    trace: &[F],
    heights: &[F],
    block_counts: &mut [u32],
    num_channels: u32,
    num_samples: u32,
    scan_start: u32,
    scan_end: u32,
    num_blocks: u32,
    lanes: u32,
    block: u32,
    polarity: u32,
) {
    let b = sample_position();
    let ch = channel_position();
    if ch < num_channels && b < num_blocks {
        let height = heights[ch as usize];
        let row = ch * num_samples;
        let mut n = 0u32;
        let mut t = first_candidate_sample(scan_start, b, lanes, block);
        let mut i = 0u32;
        while i < block && t < scan_end {
            if is_candidate::<F>(trace, row, t, height, polarity) {
                n += 1u32;
            }
            i += 1u32;
            t += lanes;
        }
        block_counts[(ch * num_blocks + b) as usize] = n;
    }
}

/// One unit per channel turns its row of `block_counts` into exclusive offsets in place and writes
/// the channel's total to `channel_totals`.
#[cube(launch)]
pub fn scan_candidate_counts_kernel(
    block_counts: &mut [u32],
    channel_totals: &mut [u32],
    num_channels: u32,
    num_blocks: u32,
) {
    let ch = ABSOLUTE_POS_X;
    if ch < num_channels {
        let row = ch * num_blocks;
        let mut acc = 0u32;
        let mut b = 0u32;
        while b < num_blocks {
            let n = block_counts[(row + b) as usize];
            block_counts[(row + b) as usize] = acc;
            acc += n;
            b += 1u32;
        }
        channel_totals[ch as usize] = acc;
    }
}

/// Each `(block, channel)` unit holding candidates writes their local sample, value and channel
/// from `channel_bases[ch] + block_offsets[ch · num_blocks + b]` on, so every channel's list is
/// contiguous (`channel_bases` has `num_channels + 1` entries).
#[cube(launch)]
pub fn write_peak_candidates_kernel<F: Float>(
    trace: &[F],
    heights: &[F],
    block_offsets: &[u32],
    channel_bases: &[u32],
    out_indices: &mut [u32],
    out_values: &mut [F],
    out_rows: &mut [u32],
    num_channels: u32,
    num_samples: u32,
    scan_start: u32,
    scan_end: u32,
    num_blocks: u32,
    lanes: u32,
    block: u32,
    polarity: u32,
) {
    let b = sample_position();
    let ch = channel_position();
    if ch < num_channels && b < num_blocks {
        let offset = block_offsets[(ch * num_blocks + b) as usize];
        let base = channel_bases[ch as usize];
        let next = if b + 1u32 < num_blocks {
            block_offsets[(ch * num_blocks + b + 1u32) as usize]
        } else {
            channel_bases[(ch + 1u32) as usize] - base
        };
        if next > offset {
            let height = heights[ch as usize];
            let row = ch * num_samples;
            let mut slot = base + offset;
            let mut t = first_candidate_sample(scan_start, b, lanes, block);
            let mut i = 0u32;
            while i < block && t < scan_end {
                if is_candidate::<F>(trace, row, t, height, polarity) {
                    out_indices[slot as usize] = t;
                    out_values[slot as usize] = trace[(row + t) as usize];
                    out_rows[slot as usize] = ch;
                    slot += 1u32;
                }
                i += 1u32;
                t += lanes;
            }
        }
    }
}
