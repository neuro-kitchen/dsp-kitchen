//! MountainSort 5 spike detection on the device (`mountainsort5/core/detect_spikes.py`), then its
//! `remove_duplicate_times`.
//!
//! A sample is a detection when it reaches the threshold (whitened units; `x ≤ −threshold` for
//! sign −1, `x ≥ threshold` for +1, `|x| ≥ threshold` for 0) and **no sample is strictly lower**
//! (in the sign's sense) on any channel within `channel_radius` µm (all channels when `None`),
//! within `±time_radius` samples (`⌈time_radius_ms · fs / 1000⌉`). Samples within `margin_left` of
//! the recording's start or `margin_right` of its end are neither detections nor compared against.
//! Detections at the same sample on different channels keep one (the lowest channel; upstream
//! keeps the first of an unstable sort), since isosplit does poorly with duplicate points.
//!
//! On the device: threshold-reaching local extrema are found and compacted
//! ([`dsp_base::peaks::find_peak_candidates_on_device`]); a detection is always one, since its
//! neighbouring samples on its own channel are not lower. One unit per candidate then scans its
//! neighbourhood window in the trace ([`fn@super::kernels::locally_exclusive_kernel`]); only the
//! kept candidates' samples and channels are downloaded. Difference: a flat run of equal samples
//! reports its first sample, where upstream reports every sample of the run (exact ties of
//! filtered floats).

use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;
use dsp_base::core::buffer;
use dsp_base::peaks::{find_peak_candidates_on_device, Polarity};
use dsp_core::compute::LaunchGeometry;
use dsp_synapse::features::ChannelNeighbourhoods;

use super::kernels::locally_exclusive_kernel;

/// Detection settings (MountainSort 5 scheme 1 defaults in [`Default`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectOptions {
    /// Whitened units.
    pub threshold: f32,
    /// −1 negative peaks, +1 positive, 0 both.
    pub sign: i8,
    pub time_radius_ms: f64,
    /// µm; `None`: every channel is a neighbour.
    pub channel_radius_um: Option<f32>,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self { threshold: 5.5, sign: -1, time_radius_ms: 0.5, channel_radius_um: None }
    }
}

impl DetectOptions {
    /// `⌈time_radius_ms · fs / 1000⌉` samples.
    pub fn time_radius(&self, sample_rate_hz: f64) -> usize {
        (self.time_radius_ms / 1000.0 * sample_rate_hz).ceil() as usize
    }

    fn polarity(&self) -> (Polarity, u32) {
        match self.sign {
            s if s < 0 => (Polarity::Negative, 0),
            s if s > 0 => (Polarity::Positive, 1),
            _ => (Polarity::Both, 2),
        }
    }
}

/// Detections, sorted by sample, one per sample.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Detections {
    /// Global samples.
    pub samples: Vec<u64>,
    pub channels: Vec<u32>,
    /// The trace's value at each detection (whitened units).
    pub values: Vec<f32>,
}

/// The neighbourhood table on the device, built once per probe.
pub struct Detector {
    opts: DetectOptions,
    neighbourhoods: ChannelNeighbourhoods,
    table: Handle,
    heights: Handle,
    channels: usize,
}

impl Detector {
    /// `positions`: `(x, y)` µm per channel.
    pub fn new(client: &Client, positions: &[[f32; 2]], opts: DetectOptions) -> Self {
        let neighbourhoods = ChannelNeighbourhoods::within_radius(positions, opts.channel_radius_um.unwrap_or(f32::INFINITY));
        let channels = positions.len();
        Self {
            table: buffer::upload(client, &neighbourhoods.table),
            heights: buffer::upload(client, &vec![opts.threshold; channels.max(1)]),
            neighbourhoods,
            opts,
            channels,
        }
    }

    /// Detections of the `[channels, samples]` device trace whose local sample lies in `emit`
    /// (module docs). `valid`: the local samples outside the recording's margins (only they are
    /// detections or compared against); `emit` should keep `time_radius` samples of context from
    /// the buffer's ends unless they are the recording's. `global_offset`: global sample of local
    /// sample 0.
    ///
    /// # Panics
    ///
    /// If the trace's channels differ from the probe's.
    pub fn detect(
        &self,
        client: &Client,
        trace: &Handle,
        samples: usize,
        valid: Range<usize>,
        emit: Range<usize>,
        global_offset: u64,
        sample_rate_hz: f64,
    ) -> Detections {
        let channels = self.channels;
        let valid = valid.start..valid.end.min(samples);
        let scan = emit.start.max(valid.start)..emit.end.min(valid.end);
        if channels == 0 || scan.is_empty() {
            return Detections::default();
        }
        let (polarity, mode) = self.opts.polarity();
        // Candidates are local extrema: a detection at the buffer's edge has no neighbour sample
        // there, so the candidate scan includes the edges (the peak finder skips them)
        let cand = find_peak_candidates_on_device::<f32>(client, trace, &self.heights, channels, samples, scan.clone(), polarity);
        let n = cand.total;
        let mut found: Vec<(u64, u32, f32)> = Vec::new();
        if n > 0 {
            let keep = buffer::empty::<u32>(client, n);
            let geom = LaunchGeometry::elementwise(client, n);
            // SAFETY: every array is passed with the length it was created with
            unsafe {
                locally_exclusive_kernel::launch::<f32>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(trace.clone(), channels * samples),
                    BufferArg::from_raw_parts(cand.indices.clone(), n),
                    BufferArg::from_raw_parts(cand.rows.clone(), n),
                    BufferArg::from_raw_parts(self.table.clone(), self.neighbourhoods.table.len()),
                    BufferArg::from_raw_parts(keep.clone(), n),
                    n as u32,
                    samples as u32,
                    self.neighbourhoods.max_neighbours as u32,
                    self.opts.time_radius(sample_rate_hz) as u32,
                    valid.start as u32,
                    valid.end as u32,
                    mode,
                );
            }
            let keep = buffer::download::<u32>(client, keep);
            let idx = buffer::download::<u32>(client, cand.indices);
            let rows = buffer::download::<u32>(client, cand.rows);
            let vals = buffer::download::<f32>(client, cand.values);
            found.extend((0..n).filter(|&i| keep[i] != 0).map(|i| (global_offset + idx[i] as u64, rows[i], vals[i])));
        }
        // The peak finder skips each buffer's first and last sample
        let host_edges = [scan.start, scan.end - 1];
        let edges: Vec<usize> = host_edges.iter().copied().filter(|&t| t == 0 || t + 1 == samples).collect();
        if !edges.is_empty() {
            found.extend(self.edge_detections(client, trace, samples, &valid, &edges, global_offset, sample_rate_hz, mode));
        }
        found.sort_unstable_by_key(|d| (d.0, d.1));
        found.dedup_by_key(|d| d.0);
        Detections {
            samples: found.iter().map(|d| d.0).collect(),
            channels: found.iter().map(|d| d.1).collect(),
            values: found.iter().map(|d| d.2).collect(),
        }
    }

    /// Detections at the buffer edges `ts` (rare: margins usually exclude them), on the host.
    #[allow(clippy::too_many_arguments)]
    fn edge_detections(
        &self,
        client: &Client,
        trace: &Handle,
        samples: usize,
        valid: &Range<usize>,
        ts: &[usize],
        global_offset: u64,
        sample_rate_hz: f64,
        mode: u32,
    ) -> Vec<(u64, u32, f32)> {
        let r = self.opts.time_radius(sample_rate_hz);
        let x = buffer::download::<f32>(client, trace.clone());
        let signed = |v: f32| match mode {
            0 => v,
            1 => -v,
            _ => -v.abs(),
        };
        let mut out = Vec::new();
        for &t in ts {
            for ch in 0..self.channels {
                let v = signed(x[ch * samples + t]);
                if v > -self.opts.threshold {
                    continue;
                }
                let window = t.saturating_sub(r).max(valid.start)..(t + r + 1).min(valid.end);
                let lower = self.neighbourhoods.of(ch).any(|nb| window.clone().any(|tt| signed(x[nb * samples + tt]) < v));
                if !lower {
                    out.push((global_offset + t as u64, ch as u32, x[ch * samples + t]));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::compute::ComputeTarget;

    /// Upstream `detect_spikes` + `remove_duplicate_times`, as written.
    fn reference(x: &[f32], m: usize, n: usize, pos: &[[f32; 2]], opts: &DetectOptions, r: usize, ml: usize, mr: usize) -> Vec<(u64, u32)> {
        let signed = |v: f32| match opts.sign {
            s if s < 0 => v,
            s if s > 0 => -v,
            _ => -v.abs(),
        };
        let radius = opts.channel_radius_um.unwrap_or(f32::INFINITY);
        let mut cand: Vec<Vec<(usize, f32)>> = vec![Vec::new(); m];
        for t in ml..n - mr {
            for ch in 0..m {
                let v = signed(x[ch * n + t]);
                if v <= -opts.threshold {
                    cand[ch].push((t, v));
                }
            }
        }
        let mut out = Vec::new();
        for ch in 0..m {
            for &(t, v) in &cand[ch] {
                let ok = (0..m)
                    .filter(|&b| (pos[ch][0] - pos[b][0]).hypot(pos[ch][1] - pos[b][1]) <= radius)
                    .all(|b| cand[b].iter().all(|&(tt, vv)| tt + r < t || tt > t + r || vv >= v));
                if ok {
                    out.push((t as u64, ch as u32));
                }
            }
        }
        out.sort_unstable();
        out.dedup_by_key(|d| d.0);
        out
    }

    #[test]
    fn matches_the_upstream_rule() {
        let Ok(target) = ComputeTarget::from_env() else { return };
        let client = target.client().expect("client");
        let (m, n, fs) = (6usize, 6000usize, 30_000.0);
        let pos: Vec<[f32; 2]> = (0..m).map(|c| [0.0, 25.0 * c as f32]).collect();
        // Noise plus spikes of varied size and sign, some overlapping across channels
        let mut state = 0x9e37_79b9u32;
        let mut x: Vec<f32> = (0..m * n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 3.0
            })
            .collect();
        for k in 0..120usize {
            let (ch, t) = (k * 7 % m, 40 + k * 47 % (n - 80));
            let a = if k % 5 == 0 { 9.0 } else { -6.0 - (k % 7) as f32 };
            for (d, w) in [(-1i32, 0.5f32), (0, 1.0), (1, 0.6)] {
                for (dc, s) in [(0i32, 1.0f32), (1, 0.7), (-1, 0.4)] {
                    let c = ch as i32 + dc;
                    if (0..m as i32).contains(&c) {
                        x[c as usize * n + (t as i32 + d) as usize] += a * w * s;
                    }
                }
            }
        }
        let trace = buffer::upload(&client, &x);
        let (ml, mr) = (20usize, 20usize);
        for sign in [-1i8, 1, 0] {
            for radius in [None, Some(30.0f32)] {
                let opts = DetectOptions { threshold: 4.0, sign, time_radius_ms: 0.5, channel_radius_um: radius };
                let r = opts.time_radius(fs);
                let want = reference(&x, m, n, &pos, &opts, r, ml, mr);
                let det = Detector::new(&client, &pos, opts);
                let got = det.detect(&client, &trace, n, ml..n - mr, 0..n, 1000, fs);
                let got: Vec<(u64, u32)> = got.samples.iter().zip(&got.channels).map(|(&s, &c)| (s - 1000, c)).collect();
                assert!(!want.is_empty(), "sign {sign}: spikes planted");
                assert_eq!(got, want, "sign {sign}, radius {radius:?}");
                // Two windows with context give the same result
                let (a, b) = (0..3000, 3000..n);
                let mut split = det.detect(&client, &trace, n, ml..n - mr, a, 1000, fs).samples;
                split.extend(det.detect(&client, &trace, n, ml..n - mr, b, 1000, fs).samples);
                assert_eq!(split, want.iter().map(|d| d.0 + 1000).collect::<Vec<_>>());
            }
        }
    }
}
