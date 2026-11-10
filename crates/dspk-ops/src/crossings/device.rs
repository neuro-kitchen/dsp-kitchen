//! Threshold crossings of a device buffer: crossings and stretch extrema found on the device, paired on the host.

use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;
use dspk_core::compute::LaunchGeometry;

use super::kernels::{column_kernel, count_crossings_kernel, stretch_extrema_kernel, write_crossings_kernel};
use super::{CrossingEvent, CrossingOptions, CrossingTracker};
use crate::core::{buffer, cast, cast_f32, to_f64, DspFloat};
use crate::error::OpsResult;
use crate::peaks::kernels::scan_candidate_counts_kernel;

/// Samples each unit of the crossing kernels scans (a few crossings per block at most on an
/// envelope; contiguous blocks keep each channel's crossings in time order).
const BLOCK: usize = 1024;

impl CrossingTracker {
    /// One window held on the device: `trace` is `[channels, samples]` of `F`, local sample 0 is
    /// sample `origin` of the run, and `scan` the local samples this window owns. Only the
    /// crossings, one extremum per stretch between them, and the window's last column are
    /// downloaded.
    pub fn push_device_window<F: DspFloat>(&mut self, client: &Client, trace: &Handle, samples: usize, scan: Range<usize>, origin: u64) {
        let channels = self.channels();
        let scan = scan.start..scan.end.min(samples);
        if channels == 0 || scan.is_empty() {
            return;
        }
        for side in 0..self.sides.len() {
            self.push_side::<F>(side, client, trace, samples, scan.clone(), origin);
        }
        let last = buffer::empty::<F>(client, channels);
        let per_channel = LaunchGeometry::per_channel(client, channels);
        // SAFETY: `trace` holds `channels · samples` values of `F`, `last` `channels`
        unsafe {
            column_kernel::launch::<F>(
                client,
                per_channel.cube_count,
                per_channel.cube_dim,
                BufferArg::from_raw_parts(trace.clone(), channels * samples),
                BufferArg::from_raw_parts(last.clone(), channels),
                channels as u32,
                samples as u32,
                (scan.end - 1) as u32,
            );
        }
        self.prev = buffer::download_prefix::<F>(client, last, channels).into_iter().map(|v| to_f64(v) as f32).collect();
    }

    /// One side's crossings and stretch extrema of a window (see [`Self::push_device_window`]).
    fn push_side<F: DspFloat>(&mut self, side: usize, client: &Client, trace: &Handle, samples: usize, scan: Range<usize>, origin: u64) {
        let channels = self.channels();
        let sign: F = cast(self.sides[side].sign as f64);
        let levels = buffer::upload(client, &cast_f32::<F>(&self.sides[side].levels));
        let releases = buffer::upload(client, &cast_f32::<F>(&self.sides[side].releases));
        let prev = buffer::upload(client, &cast_f32::<F>(&self.prev));
        let len = scan.len();
        let blocks = len.div_ceil(BLOCK);
        let tiles = LaunchGeometry::channels_samples(client, channels, blocks);
        let per_channel = LaunchGeometry::per_channel(client, channels);
        let offsets = buffer::empty::<u32>(client, channels * blocks);
        let totals = buffer::empty::<u32>(client, channels);
        // SAFETY: `trace` holds `channels · samples` values of `F`; `prev`, `levels` and `releases`
        // `channels`; the offsets and totals were just sized for `channels · blocks` and `channels`
        unsafe {
            count_crossings_kernel::launch::<F>(
                client,
                tiles.cube_count.clone(),
                tiles.cube_dim,
                BufferArg::from_raw_parts(trace.clone(), channels * samples),
                BufferArg::from_raw_parts(prev.clone(), channels),
                BufferArg::from_raw_parts(levels.clone(), channels),
                BufferArg::from_raw_parts(releases.clone(), channels),
                BufferArg::from_raw_parts(offsets.clone(), channels * blocks),
                channels as u32,
                samples as u32,
                scan.start as u32,
                scan.end as u32,
                blocks as u32,
                BLOCK as u32,
                sign,
            );
            scan_candidate_counts_kernel::launch(
                client,
                per_channel.cube_count.clone(),
                per_channel.cube_dim,
                BufferArg::from_raw_parts(offsets.clone(), channels * blocks),
                BufferArg::from_raw_parts(totals.clone(), channels),
                channels as u32,
                blocks as u32,
            );
        }
        let counts = buffer::download_prefix::<u32>(client, totals, channels);
        let mut bases = Vec::with_capacity(channels + 1);
        bases.push(0usize);
        for &n in &counts {
            bases.push(bases.last().copied().unwrap_or(0) + n as usize);
        }
        let total = bases[channels];
        let (indices, kinds) = if total == 0 {
            (Vec::new(), Vec::new())
        } else {
            let bases_u32: Vec<u32> = bases.iter().map(|&b| b as u32).collect();
            let bases_handle = buffer::upload(client, &bases_u32);
            let indices = buffer::empty::<u32>(client, total);
            let kinds = buffer::empty::<u32>(client, total);
            // SAFETY: as above; `bases_handle` holds `channels + 1` and the outputs `total` `u32`
            unsafe {
                write_crossings_kernel::launch::<F>(
                    client,
                    tiles.cube_count,
                    tiles.cube_dim,
                    BufferArg::from_raw_parts(trace.clone(), channels * samples),
                    BufferArg::from_raw_parts(prev, channels),
                    BufferArg::from_raw_parts(levels, channels),
                    BufferArg::from_raw_parts(releases, channels),
                    BufferArg::from_raw_parts(offsets, channels * blocks),
                    BufferArg::from_raw_parts(bases_handle, channels + 1),
                    BufferArg::from_raw_parts(indices.clone(), total),
                    BufferArg::from_raw_parts(kinds.clone(), total),
                    channels as u32,
                    samples as u32,
                    scan.start as u32,
                    scan.end as u32,
                    blocks as u32,
                    BLOCK as u32,
                    sign,
                );
            }
            (buffer::download_prefix::<u32>(client, indices, total), buffer::download_prefix::<u32>(client, kinds, total))
        };

        // The stretches: before each channel's first crossing, then after each crossing
        let stretches = total + channels;
        let (mut s_ch, mut s_start, mut s_end) = (Vec::with_capacity(stretches), Vec::with_capacity(stretches), Vec::with_capacity(stretches));
        for ch in 0..channels {
            let mut start = scan.start as u32;
            for &t in &indices[bases[ch]..bases[ch + 1]] {
                s_ch.push(ch as u32);
                s_start.push(start);
                s_end.push(t);
                start = t;
            }
            s_ch.push(ch as u32);
            s_start.push(start);
            s_end.push(scan.end as u32);
        }
        let values = buffer::empty::<F>(client, stretches);
        let at = buffer::empty::<u32>(client, stretches);
        let per_stretch = LaunchGeometry::per_channel(client, stretches);
        // SAFETY: the stretch arrays and outputs hold `stretches` elements
        unsafe {
            stretch_extrema_kernel::launch::<F>(
                client,
                per_stretch.cube_count,
                per_stretch.cube_dim,
                BufferArg::from_raw_parts(trace.clone(), channels * samples),
                BufferArg::from_raw_parts(buffer::upload(client, &s_ch), stretches),
                BufferArg::from_raw_parts(buffer::upload(client, &s_start), stretches),
                BufferArg::from_raw_parts(buffer::upload(client, &s_end), stretches),
                BufferArg::from_raw_parts(values.clone(), stretches),
                BufferArg::from_raw_parts(at.clone(), stretches),
                stretches as u32,
                samples as u32,
                sign,
            );
        }
        let values: Vec<f32> = buffer::download_prefix::<F>(client, values, stretches).into_iter().map(|v| to_f64(v) as f32).collect();
        let at = buffer::download_prefix::<u32>(client, at, stretches);

        let mut crossings = Vec::new();
        let mut extrema = Vec::new();
        for ch in 0..channels {
            let r = bases[ch]..bases[ch + 1];
            crossings.clear();
            crossings.extend(r.clone().map(|i| (indices[i], kinds[i])));
            // Channel `ch`'s stretches start at `bases[ch] + ch`
            let s = bases[ch] + ch..bases[ch + 1] + ch + 1;
            extrema.clear();
            extrema.extend(s.map(|i| (values[i], at[i])));
            self.take(side, ch, origin, &crossings, &extrema);
        }
    }
}

/// Threshold-crossing events of every channel of the `[channels, samples]` device buffer `trace` (see the module docs).
pub fn detect_crossings_on_device<F: DspFloat>(client: &Client, trace: &Handle, channels: usize, samples: usize, options: &CrossingOptions) -> OpsResult<Vec<CrossingEvent>> {
    let mut tracker = CrossingTracker::new(channels, options)?;
    tracker.push_device_window::<F>(client, trace, samples, 0..samples, 0);
    Ok(tracker.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crossings::{detect_crossings, Direction};

    /// A noisy envelope with bursts on 3 channels, more samples than one block.
    fn envelope(channels: usize, samples: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; channels * samples];
        let mut state = 0x2545_f491u32;
        for c in 0..channels {
            for t in 0..samples {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (state >> 8) as f32 / (1u32 << 24) as f32;
                let burst = if (t / 700 + c) % 3 == 0 { 6.0 + ((t % 700) as f32 * 0.01).sin() * 2.0 } else { 0.0 };
                x[c * samples + t] = noise + burst;
            }
        }
        x
    }

    fn matches_host(client: &Client) {
        let (channels, samples) = (3usize, 5000usize);
        let x = envelope(channels, samples);
        // Both: bursts up on even samples' stretches and down on odd ones (every 700 samples)
        let mixed: Vec<f32> = x.iter().enumerate().map(|(i, v)| if (i % samples) / 700 % 2 == 0 { *v } else { -v }).collect();
        for direction in [Direction::Rising, Direction::Falling, Direction::Both] {
            let (s, data): (f32, Vec<f32>) = match direction {
                Direction::Rising => (1.0, x.clone()),
                Direction::Falling => (-1.0, x.iter().map(|v| -v).collect()),
                Direction::Both => (1.0, mixed.clone()),
            };
            let trace = buffer::upload(client, &data);
            let mut o = CrossingOptions::new(s * 4.0);
            o.releases = vec![s * 2.0];
            o.direction = direction;
            o.min_duration_samples = 50;
            let host = detect_crossings(&data, channels, samples, &o).unwrap();
            assert!(host.len() > 6, "{direction:?}: {}", host.len());
            assert_eq!(detect_crossings_on_device::<f32>(client, &trace, channels, samples, &o).unwrap(), host, "{direction:?}");
            // Windows of 1500 samples (with a 10-sample left halo, as pipeline windows have)
            let mut t = CrossingTracker::new(channels, &o).unwrap();
            let mut start = 0;
            while start < samples {
                let read = start.saturating_sub(10);
                let end = (start + 1500).min(samples);
                let n = end - read;
                let rows: Vec<f32> = (0..channels).flat_map(|c| data[c * samples + read..c * samples + end].to_vec()).collect();
                let w = buffer::upload(client, &rows);
                t.push_device_window::<f32>(client, &w, n, start - read..n, read as u64);
                start = end;
            }
            assert_eq!(t.finish(), host, "{direction:?} in windows");
        }
    }

    runtime_test!(device_crossings_match_host, matches_host);
}
