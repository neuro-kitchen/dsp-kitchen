use cubecl::prelude::*;
use dsp_base::geometry::LaunchGeometry;
use crate::detection::SpikeEvent;

/// Sentinel for "no spike kept" in the per-channel last-spike output.
pub const NO_SPIKE: u32 = u32::MAX;

/// CubeCL kernel for multi-channel negative trough threshold detection with refractory suppression.
///
/// Parallelized with 1 GPU thread per channel (`ABSOLUTE_POS_X`), scanning sequentially across the
/// valid interior window `[valid_start, valid_end)` directly inside the in-VRAM filtered buffer.
///
/// - `first_allowed[ch]`: earliest local sample a crossing may be kept at (refractory carried over
///   from the previous window), so windowed detection equals one pass over the whole recording.
/// - `out_counts[ch]`: number of crossings kept, **including** those beyond
///   `max_spikes_per_channel` (only the first `max_spikes_per_channel` are written).
/// - `out_last[ch]`: local sample of the last kept crossing, or [`NO_SPIKE`].
#[cube(launch)]
pub fn detect_channel_troughs_kernel(
    trace: &Array<f32>,
    channel_sigmas: &Array<f32>,
    first_allowed: &Array<u32>,
    out_sample_indices: &mut Array<u32>,
    out_amplitudes: &mut Array<f32>,
    out_counts: &mut Array<u32>,
    out_last: &mut Array<u32>,
    num_channels: u32,
    num_samples: u32,
    valid_start: u32,
    valid_end: u32,
    threshold_factor: f32,
    refractory_samples: u32,
    max_spikes_per_channel: u32,
) {
    let ch = ABSOLUTE_POS_X;

    if ch < num_channels {
        let sigma = channel_sigmas[ch as usize];
        let mut count: u32 = 0u32;
        let mut last_spike_t: u32 = 0xFFFF_FFFFu32; // NO_SPIKE

        if sigma > 0.0f32 {
            let thresh = -threshold_factor * sigma;
            let ch_offset = ch * num_samples;
            let out_offset = ch * max_spikes_per_channel;

            let start_t = u32::max(valid_start, 1u32);
            let end_t = u32::min(valid_end, num_samples - 1u32);
            let mut next_allowed = first_allowed[ch as usize];
            let mut t: u32 = start_t;

            while t < end_t {
                let idx = (ch_offset + t) as usize;
                let val = trace[idx];
                let prev = trace[(ch_offset + t - 1u32) as usize];
                let next = trace[(ch_offset + t + 1u32) as usize];

                if val < thresh && val < prev && val <= next && t >= next_allowed {
                    if count < max_spikes_per_channel {
                        let write_idx = (out_offset + count) as usize;
                        out_sample_indices[write_idx] = t;
                        out_amplitudes[write_idx] = val;
                    }
                    count += 1u32;
                    last_spike_t = t;
                    next_allowed = t + refractory_samples + 1u32;
                }
                t += 1u32;
            }
        }

        out_counts[ch as usize] = count;
        out_last[ch as usize] = last_spike_t;
    }
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

/// Host-side dispatcher executing [`detect_channel_troughs_kernel`] directly on an in-VRAM
/// filtered trace handle and downloading only the compact spike events.
///
/// Scans local samples `[valid_start, valid_end)` of a buffer whose local sample 0 is global sample
/// `global_offset`; returned events carry **global** sample indices. With `carry`, the refractory
/// period continues from the previous window and is updated for the next one. If a channel holds
/// more than `max_spikes_per_channel` crossings, detection is repeated with a larger buffer (no
/// crossing is lost).
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
    max_spikes_per_channel: usize,
    carry: Option<&mut DetectionCarry>,
) -> Vec<SpikeEvent> {
    if channels == 0 || samples < 3 || valid_start >= valid_end {
        return Vec::new();
    }

    let first_allowed: Vec<u32> = (0..channels)
        .map(|ch| {
            let last = carry.as_ref().and_then(|c| c.last_spike.get(ch).copied().flatten());
            match last {
                None => 0,
                Some(g) => (g + refractory_samples as u64 + 1).saturating_sub(global_offset).min(u32::MAX as u64) as u32,
            }
        })
        .collect();
    let first_allowed_handle = client.create_from_slice(u32::as_bytes(&first_allowed));

    let mut max_spikes = max_spikes_per_channel.max(1);
    let (counts, last, indices, amps) = loop {
        let total_spike_slots = channels * max_spikes;
        let out_indices_handle = client.empty(total_spike_slots * std::mem::size_of::<u32>());
        let out_amps_handle = client.empty(total_spike_slots * std::mem::size_of::<f32>());
        let out_counts_handle = client.empty(channels * std::mem::size_of::<u32>());
        let out_last_handle = client.empty(channels * std::mem::size_of::<u32>());

        let geom = LaunchGeometry::per_channel(channels);
        unsafe {
            detect_channel_troughs_kernel::launch::<R>(
                client,
                geom.cube_count,
                geom.cube_dim,
                ArrayArg::from_raw_parts(trace_handle.clone(), channels * samples),
                ArrayArg::from_raw_parts(sigmas_handle.clone(), channels),
                ArrayArg::from_raw_parts(first_allowed_handle.clone(), channels),
                ArrayArg::from_raw_parts(out_indices_handle.clone(), total_spike_slots),
                ArrayArg::from_raw_parts(out_amps_handle.clone(), total_spike_slots),
                ArrayArg::from_raw_parts(out_counts_handle.clone(), channels),
                ArrayArg::from_raw_parts(out_last_handle.clone(), channels),
                channels as u32,
                samples as u32,
                valid_start as u32,
                valid_end as u32,
                threshold_factor,
                refractory_samples as u32,
                max_spikes as u32,
            );
        }

        let counts = u32::from_bytes(&client.read_one(out_counts_handle).expect("VRAM read counts")).to_vec();
        let needed = counts.iter().copied().max().unwrap_or(0) as usize;
        if needed > max_spikes {
            tracing::warn!(needed, capacity = max_spikes, "spike buffer overflow; re-running detection with a larger buffer");
            max_spikes = needed;
            continue;
        }
        let last = u32::from_bytes(&client.read_one(out_last_handle).expect("VRAM read last")).to_vec();
        if counts.iter().all(|&c| c == 0) {
            break (counts, last, Vec::new(), Vec::new());
        }
        let indices = u32::from_bytes(&client.read_one(out_indices_handle).expect("VRAM read indices")).to_vec();
        let amps = f32::from_bytes(&client.read_one(out_amps_handle).expect("VRAM read amps")).to_vec();
        break (counts, last, indices, amps);
    };

    if let Some(carry) = carry {
        carry.last_spike.resize(channels, None);
        for (slot, &l) in carry.last_spike.iter_mut().zip(&last) {
            if l != NO_SPIKE {
                *slot = Some(global_offset + l as u64);
            }
        }
    }

    let mut events = Vec::with_capacity(counts.iter().map(|&c| c as usize).sum());
    for (ch, &cnt) in counts.iter().enumerate() {
        let base = ch * max_spikes;
        for i in 0..cnt as usize {
            events.push(SpikeEvent {
                channel_id: ch,
                sample_index: global_offset + indices[base + i] as u64,
                peak_amplitude_uv: amps[base + i],
            });
        }
    }

    events.sort_by_key(|s| (s.sample_index, s.channel_id));
    events
}
