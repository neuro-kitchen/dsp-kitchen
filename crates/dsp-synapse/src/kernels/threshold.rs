use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{AutotuneOutput, LocalTuner, Tunable, TunableSet, local_tuner};
use dsp_core::compute::{LaunchGeometry, channel_position, sample_position};
use dsp_core::compute::tune::{size_class, tune_id};
use crate::detection::SpikeEvent;

/// Samples scanned per unit tried by the autotuner (work per unit against number of units; the
/// fastest depends on the device and is measured, see [`dsp_core::compute::tune`]).
const BLOCK_CANDIDATES: [u32; 5] = [16, 64, 256, 1024, 4096];

/// First sample scanned by unit `b`; it then steps by `lanes`. Units are grouped `lanes` at a
/// time (the runtime's plane size, see [`LaunchGeometry::plane_lanes`]) and a group's lanes
/// interleave over its `lanes · block` samples, so the units of a plane read neighbouring addresses.
/// With one-unit planes every unit scans `block` contiguous samples.
#[cube]
fn first_candidate_sample(scan_start: u32, b: u32, lanes: u32, block: u32) -> u32 {
    let group = b / lanes;
    scan_start + group * lanes * block + (b - group * lanes)
}

/// Whether local sample `t` of the row starting at `row` is a negative trough below `thresh`
/// (`x[t] < x[t-1]`, `x[t] <= x[t+1]`).
#[cube]
fn is_trough(trace: &Array<f32>, row: u32, t: u32, thresh: f32) -> bool {
    let val = trace[(row + t) as usize];
    val < thresh && val < trace[(row + t - 1u32) as usize] && val <= trace[(row + t + 1u32) as usize]
}

/// First pass of the candidate compaction: one unit per `(block, channel)` (block =
/// [`sample_position`], channel = [`channel_position`]) counts the troughs below `-threshold_factor · σ` among its
/// `block` samples (see [`first_candidate_sample`]) into `block_counts[ch · num_blocks + b]`.
#[cube(launch)]
pub fn count_trough_candidates_kernel(
    trace: &Array<f32>,
    channel_sigmas: &Array<f32>,
    block_counts: &mut Array<u32>,
    num_channels: u32,
    num_samples: u32,
    scan_start: u32,
    scan_end: u32,
    num_blocks: u32,
    lanes: u32,
    block: u32,
    threshold_factor: f32,
) {
    let b = sample_position();
    let ch = channel_position();
    if ch < num_channels && b < num_blocks {
        let sigma = channel_sigmas[ch as usize];
        let mut n = 0u32;
        if sigma > 0.0f32 {
            let thresh = -threshold_factor * sigma;
            let row = ch * num_samples;
            let mut t = first_candidate_sample(scan_start, b, lanes, block);
            let mut i: u32 = 0u32;
            while i < block && t < scan_end {
                if is_trough(trace, row, t, thresh) {
                    n += 1u32;
                }
                i += 1u32;
                t += lanes;
            }
        }
        block_counts[(ch * num_blocks + b) as usize] = n;
    }
}

/// Second pass: one unit per channel turns its row of `block_counts` into exclusive offsets in
/// place and writes the channel's total to `channel_totals`.
#[cube(launch)]
pub fn scan_candidate_counts_kernel(
    block_counts: &mut Array<u32>,
    channel_totals: &mut Array<u32>,
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

/// Third pass: each `(block, channel)` unit holding candidates writes their local sample and
/// amplitude from `channel_bases[ch] + block_offsets[ch · num_blocks + b]` on, so every channel's
/// list is contiguous (`channel_bases` has `num_channels + 1` entries).
#[cube(launch)]
pub fn write_trough_candidates_kernel(
    trace: &Array<f32>,
    channel_sigmas: &Array<f32>,
    block_offsets: &Array<u32>,
    channel_bases: &Array<u32>,
    out_sample_indices: &mut Array<u32>,
    out_amplitudes: &mut Array<f32>,
    num_channels: u32,
    num_samples: u32,
    scan_start: u32,
    scan_end: u32,
    num_blocks: u32,
    lanes: u32,
    block: u32,
    threshold_factor: f32,
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
            let thresh = -threshold_factor * channel_sigmas[ch as usize];
            let row = ch * num_samples;
            let mut slot = base + offset;
            let mut t = first_candidate_sample(scan_start, b, lanes, block);
            let mut i: u32 = 0u32;
            while i < block && t < scan_end {
                if is_trough(trace, row, t, thresh) {
                    out_sample_indices[slot as usize] = t;
                    out_amplitudes[slot as usize] = trace[(row + t) as usize];
                    slot += 1u32;
                }
                i += 1u32;
                t += lanes;
            }
        }
    }
}

/// Inputs of the count and scan passes (cloned per autotune candidate).
#[derive(Clone)]
struct CountInputs<R: Runtime> {
    client: ComputeClient<R>,
    trace: Handle,
    sigmas: Handle,
    channels: usize,
    samples: usize,
    scan: (usize, usize),
    threshold_factor: f32,
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

/// Count and scan passes with `block` samples per unit.
fn count_candidates<R: Runtime>(i: &CountInputs<R>, block: u32) -> CountPass {
    let (client, channels) = (&i.client, i.channels);
    let lanes = LaunchGeometry::plane_lanes(client) as usize;
    let blocks = (i.scan.1 - i.scan.0).div_ceil(lanes * block as usize) * lanes;
    let tiles = LaunchGeometry::channels_samples(client, channels, blocks);
    let per_channel = LaunchGeometry::per_channel(client, channels);
    let offsets = client.empty(channels * blocks * std::mem::size_of::<u32>());
    let totals = client.empty(channels * std::mem::size_of::<u32>());
    unsafe {
        count_trough_candidates_kernel::launch::<R>(
            client,
            tiles.cube_count,
            tiles.cube_dim,
            ArrayArg::from_raw_parts(i.trace.clone(), channels * i.samples),
            ArrayArg::from_raw_parts(i.sigmas.clone(), channels),
            ArrayArg::from_raw_parts(offsets.clone(), channels * blocks),
            channels as u32,
            i.samples as u32,
            i.scan.0 as u32,
            i.scan.1 as u32,
            blocks as u32,
            lanes as u32,
            block,
            i.threshold_factor,
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

/// [`count_candidates`] with the block length CubeCL's autotuner found fastest for this device and
/// problem size.
fn tuned_count<R: Runtime>(inputs: CountInputs<R>) -> CountPass {
    static TUNER: LocalTuner<String, String> = local_tuner!("trough-candidates");
    let set = TUNER.init(|| {
        let key = |i: &CountInputs<R>| format!("c{}-s{}", size_class(i.channels), size_class(i.scan.1 - i.scan.0));
        let set: TunableSet<String, CountInputs<R>, CountPass> = TunableSet::new_cloning_inputs(key);
        BLOCK_CANDIDATES.iter().fold(set, |set, &block| {
            set.with(Tunable::new(&format!("block{block}"), move |i: CountInputs<R>| Ok::<_, String>(count_candidates(&i, block))))
        })
    });
    let client = inputs.client.clone();
    TUNER.execute(&tune_id(&client), &client, set, inputs)
}

/// Per-channel refractory state carried between consecutive detection windows.
#[derive(Debug, Clone, Default)]
pub struct DetectionCarry {
    /// Global sample of the last kept crossing per channel.
    last_spike: Vec<Option<u64>>,
}

impl DetectionCarry {
    pub fn new(channels: usize) -> Self {
        Self { last_spike: vec![None; channels] }
    }
}

/// Host-side dispatcher finding threshold crossings directly on an in-VRAM filtered trace handle.
///
/// Troughs are found and compacted per channel on the device (count, scan, write passes), only the
/// compact candidate lists are downloaded, and the refractory period is applied on the host: along
/// each channel a crossing is kept when it lies more than `refractory_samples` after the previously
/// kept one.
///
/// Scans local samples `[valid_start, valid_end)` of a buffer whose local sample 0 is global sample
/// `global_offset`; returned events carry **global** sample indices. With `carry`, the refractory
/// period continues from the previous window and is updated for the next one.
#[allow(clippy::too_many_arguments)]
pub fn execute_detect_spikes_in_vram<R: Runtime>(
    client: &ComputeClient<R>,
    trace_handle: &cubecl::server::Handle,
    sigmas_handle: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    valid_start: usize,
    valid_end: usize,
    global_offset: u64,
    threshold_factor: f32,
    refractory_samples: usize,
    carry: Option<&mut DetectionCarry>,
) -> Vec<SpikeEvent> {
    let scan = valid_start.max(1)..valid_end.min(samples.saturating_sub(1));
    if channels == 0 || scan.is_empty() {
        return Vec::new();
    }
    let pass = tuned_count(CountInputs {
        client: client.clone(),
        trace: trace_handle.clone(),
        sigmas: sigmas_handle.clone(),
        channels,
        samples,
        scan: (scan.start, scan.end),
        threshold_factor,
    });
    let (lanes, blocks) = (LaunchGeometry::plane_lanes(client) as usize, pass.blocks);
    let tiles = LaunchGeometry::channels_samples(client, channels, blocks);
    let trace = || unsafe { ArrayArg::from_raw_parts(trace_handle.clone(), channels * samples) };
    let sigmas = || unsafe { ArrayArg::from_raw_parts(sigmas_handle.clone(), channels) };
    let (offsets_handle, totals_handle) = (pass.offsets, pass.totals);

    let totals = u32::from_bytes(&client.read_one(totals_handle).expect("VRAM read totals")).to_vec();
    let mut bases = Vec::with_capacity(channels + 1);
    bases.push(0u32);
    for &n in &totals {
        bases.push(bases.last().unwrap() + n);
    }
    let num_candidates = bases[channels] as usize;

    let (mut indices, mut amps) = if num_candidates == 0 {
        (Vec::new(), Vec::new())
    } else {
        let bases_handle = client.create_from_slice(u32::as_bytes(&bases));
        let out_indices_handle = client.empty(num_candidates * std::mem::size_of::<u32>());
        let out_amps_handle = client.empty(num_candidates * std::mem::size_of::<f32>());
        unsafe {
            write_trough_candidates_kernel::launch::<R>(
                client,
                tiles.cube_count,
                tiles.cube_dim,
                trace(),
                sigmas(),
                ArrayArg::from_raw_parts(offsets_handle, channels * blocks),
                ArrayArg::from_raw_parts(bases_handle, channels + 1),
                ArrayArg::from_raw_parts(out_indices_handle.clone(), num_candidates),
                ArrayArg::from_raw_parts(out_amps_handle.clone(), num_candidates),
                channels as u32,
                samples as u32,
                scan.start as u32,
                scan.end as u32,
                blocks as u32,
                lanes as u32,
                pass.block,
                threshold_factor,
            );
        }
        let indices = u32::from_bytes(&client.read_one(out_indices_handle).expect("VRAM read indices")).to_vec();
        let amps = f32::from_bytes(&client.read_one(out_amps_handle).expect("VRAM read amps")).to_vec();
        (indices, amps)
    };

    let mut last_spike: Vec<Option<u64>> = match &carry {
        Some(c) => (0..channels).map(|ch| c.last_spike.get(ch).copied().flatten()).collect(),
        None => vec![None; channels],
    };

    let mut events = Vec::new();
    let mut candidates: Vec<(u32, f32)> = Vec::new();
    for ch in 0..channels {
        // Interleaved plane lanes leave each channel's list unordered in time.
        let range = bases[ch] as usize..bases[ch + 1] as usize;
        candidates.clear();
        candidates.extend(range.clone().map(|i| (indices[i], amps[i])));
        candidates.sort_unstable_by_key(|&(t, _)| t);
        for (k, (t, a)) in range.zip(candidates.iter().copied()) {
            indices[k] = t;
            amps[k] = a;
        }
        let mut next_allowed = last_spike[ch].map_or(0, |g| g + refractory_samples as u64 + 1);
        for i in bases[ch] as usize..bases[ch + 1] as usize {
            let global = global_offset + indices[i] as u64;
            if global >= next_allowed {
                events.push(SpikeEvent { channel_id: ch, sample_index: global, peak_amplitude_uv: amps[i] });
                last_spike[ch] = Some(global);
                next_allowed = global + refractory_samples as u64 + 1;
            }
        }
    }

    if let Some(carry) = carry {
        carry.last_spike = last_spike;
    }

    events.sort_by_key(|s| (s.sample_index, s.channel_id));
    events
}
