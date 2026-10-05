//! Candidate peaks of every channel of a device buffer, compacted on the device so only the
//! candidates are downloaded.

use std::ops::Range;

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{AutotuneOutput, LocalTuner, Tunable, TunableSet, local_tuner};
use dsp_core::compute::LaunchGeometry;
use dsp_core::compute::tune::{size_class, tune_id};

use super::host::Polarity;
use super::kernels::{
    count_peak_candidates_kernel, scan_candidate_counts_kernel, write_peak_candidates_kernel, POLARITY_BOTH,
    POLARITY_NEGATIVE, POLARITY_POSITIVE,
};
use crate::core::{buffer, DspFloat};

/// Samples scanned per unit tried by the autotuner (work per unit against number of units; the
/// fastest depends on the device and is measured, see [`dsp_core::compute::tune`]).
const BLOCK_CANDIDATES: [u32; 5] = [16, 64, 256, 1024, 4096];

fn polarity_id(p: Polarity) -> u32 {
    match p {
        Polarity::Positive => POLARITY_POSITIVE,
        Polarity::Negative => POLARITY_NEGATIVE,
        Polarity::Both => POLARITY_BOTH,
    }
}

/// Candidate peaks per channel, each channel's ascending in sample.
#[derive(Debug, Clone, PartialEq)]
pub struct PeakCandidates<F> {
    /// `channels + 1` offsets: channel `c` owns `bases[c]..bases[c + 1]`.
    pub bases: Vec<usize>,
    /// Local sample of each candidate (in the scanned buffer).
    pub indices: Vec<u32>,
    /// Signal value at each candidate.
    pub values: Vec<F>,
}

impl<F> PeakCandidates<F> {
    /// No candidates on `channels` channels.
    pub fn empty(channels: usize) -> Self {
        Self { bases: vec![0; channels + 1], indices: Vec::new(), values: Vec::new() }
    }

    /// Samples and values of channel `ch`'s candidates.
    pub fn channel(&self, ch: usize) -> (&[u32], &[F]) {
        let r = self.bases[ch]..self.bases[ch + 1];
        (&self.indices[r.clone()], &self.values[r])
    }

    pub fn num_channels(&self) -> usize {
        self.bases.len().saturating_sub(1)
    }
}

/// Inputs of the count and scan passes (cloned per autotune candidate).
#[derive(Clone)]
struct CountInputs<R: Runtime> {
    client: ComputeClient<R>,
    trace: Handle,
    heights: Handle,
    channels: usize,
    samples: usize,
    scan: (usize, usize),
    polarity: u32,
}

/// Result of the count and scan passes for one block length.
struct CountPass {
    block: u32,
    blocks: usize,
    /// Exclusive candidate offsets `[channels, blocks]`.
    offsets: Handle,
    /// Candidates per channel.
    totals: Handle,
}

impl AutotuneOutput for CountPass {}

fn count_candidates<R: Runtime, F: DspFloat>(i: &CountInputs<R>, block: u32) -> CountPass {
    let (client, channels) = (&i.client, i.channels);
    let lanes = LaunchGeometry::plane_lanes(client) as usize;
    let blocks = (i.scan.1 - i.scan.0).div_ceil(lanes * block as usize) * lanes;
    let tiles = LaunchGeometry::channels_samples(client, channels, blocks);
    let per_channel = LaunchGeometry::per_channel(client, channels);
    let offsets = buffer::empty::<R, u32>(client, channels * blocks);
    let totals = buffer::empty::<R, u32>(client, channels);
    // SAFETY: `trace` holds `channels · samples` and `heights` `channels` values of `F`; the
    // offsets and totals buffers were just sized for `channels · blocks` and `channels` `u32`
    unsafe {
        count_peak_candidates_kernel::launch::<F, R>(
            client,
            tiles.cube_count,
            tiles.cube_dim,
            ArrayArg::from_raw_parts(i.trace.clone(), channels * i.samples),
            ArrayArg::from_raw_parts(i.heights.clone(), channels),
            ArrayArg::from_raw_parts(offsets.clone(), channels * blocks),
            channels as u32,
            i.samples as u32,
            i.scan.0 as u32,
            i.scan.1 as u32,
            blocks as u32,
            lanes as u32,
            block,
            i.polarity,
        );
        scan_candidate_counts_kernel::launch::<R>(
            client,
            per_channel.cube_count,
            per_channel.cube_dim,
            ArrayArg::from_raw_parts(offsets.clone(), channels * blocks),
            ArrayArg::from_raw_parts(totals.clone(), channels),
            channels as u32,
            blocks as u32,
        );
    }
    CountPass { block, blocks, offsets, totals }
}

/// [`count_candidates`] with the block length CubeCL's autotuner found fastest for this device,
/// element type and problem size.
fn tuned_count<R: Runtime, F: DspFloat>(inputs: CountInputs<R>) -> CountPass {
    static TUNER: LocalTuner<String, String> = local_tuner!("peak-candidates");
    let set = TUNER.init(|| {
        let key = |i: &CountInputs<R>| format!("{}-c{}-s{}", F::type_name(), size_class(i.channels), size_class(i.scan.1 - i.scan.0));
        let set: TunableSet<String, CountInputs<R>, CountPass> = TunableSet::new_cloning_inputs(key);
        BLOCK_CANDIDATES.iter().fold(set, |set, &block| {
            set.with(Tunable::new(&format!("block{block}"), move |i: CountInputs<R>| Ok::<_, String>(count_candidates::<R, F>(&i, block))))
        })
    });
    let client = inputs.client.clone();
    TUNER.execute(&tune_id(&client), &client, set, inputs)
}

/// Candidate peaks of the `[channels, samples]` buffer `trace` within local samples `scan`
/// (clamped to `1..samples − 1`): local extrema of `polarity` whose signed value reaches the
/// channel's entry of `heights` (`[channels]` of `F`; `+∞` skips a channel). Strict on the left
/// (`x[t−1] < x[t] ≥ x[t+1]`), so a flat peak reports its first sample, where
/// [`super::find_peaks`] reports its middle one.
///
/// Only the conditions that need no neighbouring peak run on the device; apply `distance`
/// ([`super::select_by_distance`]) or other conditions to the candidates on the host.
pub fn find_peak_candidates<R: Runtime, F: DspFloat>(
    client: &ComputeClient<R>,
    trace: &Handle,
    heights: &Handle,
    channels: usize,
    samples: usize,
    scan: Range<usize>,
    polarity: Polarity,
) -> PeakCandidates<F> {
    let scan = scan.start.max(1)..scan.end.min(samples.saturating_sub(1));
    if channels == 0 || scan.is_empty() {
        return PeakCandidates::empty(channels);
    }
    let polarity = polarity_id(polarity);
    let pass = tuned_count::<R, F>(CountInputs {
        client: client.clone(),
        trace: trace.clone(),
        heights: heights.clone(),
        channels,
        samples,
        scan: (scan.start, scan.end),
        polarity,
    });

    let totals = buffer::download::<R, u32>(client, pass.totals);
    let mut bases = Vec::with_capacity(channels + 1);
    bases.push(0usize);
    for &n in totals.iter().take(channels) {
        bases.push(bases.last().copied().unwrap_or(0) + n as usize);
    }
    let total = bases[channels];
    if total == 0 {
        return PeakCandidates::empty(channels);
    }

    let lanes = LaunchGeometry::plane_lanes(client);
    let tiles = LaunchGeometry::channels_samples(client, channels, pass.blocks);
    let bases_u32: Vec<u32> = bases.iter().map(|&b| b as u32).collect();
    let bases_handle = buffer::upload(client, &bases_u32);
    let out_indices = buffer::empty::<R, u32>(client, total);
    let out_values = buffer::empty::<R, F>(client, total);
    // SAFETY: buffers sized above (`trace` and `heights` as in `count_candidates`)
    unsafe {
        write_peak_candidates_kernel::launch::<F, R>(
            client,
            tiles.cube_count,
            tiles.cube_dim,
            ArrayArg::from_raw_parts(trace.clone(), channels * samples),
            ArrayArg::from_raw_parts(heights.clone(), channels),
            ArrayArg::from_raw_parts(pass.offsets, channels * pass.blocks),
            ArrayArg::from_raw_parts(bases_handle, channels + 1),
            ArrayArg::from_raw_parts(out_indices.clone(), total),
            ArrayArg::from_raw_parts(out_values.clone(), total),
            channels as u32,
            samples as u32,
            scan.start as u32,
            scan.end as u32,
            pass.blocks as u32,
            lanes,
            pass.block,
            polarity,
        );
    }
    let mut indices = buffer::download::<R, u32>(client, out_indices);
    let mut values = buffer::download::<R, F>(client, out_values);
    indices.truncate(total);
    values.truncate(total);

    // Interleaved plane lanes leave each channel's list unordered in time
    let mut pairs: Vec<(u32, F)> = Vec::new();
    for ch in 0..channels {
        let r = bases[ch]..bases[ch + 1];
        pairs.clear();
        pairs.extend(r.clone().map(|i| (indices[i], values[i])));
        pairs.sort_unstable_by_key(|&(t, _)| t);
        for (k, (t, v)) in r.zip(pairs.iter().copied()) {
            indices[k] = t;
            values[k] = v;
        }
    }
    PeakCandidates { bases, indices, values }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peaks::{find_peaks, PeakOptions, Interval};

    fn matches_host<R: Runtime>(client: &ComputeClient<R>) {
        let (channels, samples) = (3usize, 4000usize);
        let mut x = vec![0.0f32; channels * samples];
        let mut state = 0x1234_5678u32;
        for v in x.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
        }
        let heights = [0.5f32, 0.9, f32::INFINITY];
        let trace = buffer::upload(client, &x);
        let h = buffer::upload(client, &heights);
        for polarity in [Polarity::Positive, Polarity::Negative, Polarity::Both] {
            let c = find_peak_candidates::<R, f32>(client, &trace, &h, channels, samples, 0..samples, polarity);
            for ch in 0..channels {
                let row = &x[ch * samples..(ch + 1) * samples];
                let host = find_peaks(row, polarity, &PeakOptions { height: Interval::at_least(heights[ch]), ..Default::default() });
                let (idx, vals) = c.channel(ch);
                let idx: Vec<usize> = idx.iter().map(|&t| t as usize).collect();
                assert_eq!(idx, host.indices, "channel {ch}, {polarity:?}");
                assert!(vals.iter().zip(&idx).all(|(&v, &t)| v == row[t]));
            }
        }
    }

    runtime_test!(device_candidates_match_host_find_peaks, matches_host);
}
