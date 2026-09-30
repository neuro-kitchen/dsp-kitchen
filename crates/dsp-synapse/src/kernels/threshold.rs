use cubecl::prelude::*;
use dsp_base::geometry::LaunchGeometry;
use crate::detection::SpikeEvent;

/// CubeCL kernel for multi-channel negative trough threshold detection with refractory suppression.
///
/// Parallelized with 1 GPU thread per channel (`ABSOLUTE_POS_X`), scanning sequentially across the
/// valid interior window `[valid_start, valid_end)` directly inside the in-VRAM filtered buffer.
#[cube(launch)]
pub fn detect_channel_troughs_kernel(
    trace: &Array<f32>,
    channel_sigmas: &Array<f32>,
    out_sample_indices: &mut Array<u32>,
    out_amplitudes: &mut Array<f32>,
    out_counts: &mut Array<u32>,
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

        if sigma > 0.0f32 {
            let thresh = -threshold_factor * sigma;
            let ch_offset = ch * num_samples;
            let out_offset = ch * max_spikes_per_channel;

            let start_t = u32::max(valid_start, 1u32);
            let end_t = u32::min(valid_end, num_samples - 1u32);

            let mut has_prev: u32 = 0u32;
            let mut last_spike_t: u32 = 0u32;
            let mut t: u32 = start_t;

            while t < end_t {
                let idx = (ch_offset + t) as usize;
                let val = trace[idx];
                let prev = trace[(ch_offset + t - 1u32) as usize];
                let next = trace[(ch_offset + t + 1u32) as usize];

                if val < thresh && val < prev && val <= next {
                    if has_prev == 0u32 || t > last_spike_t + refractory_samples {
                        if count < max_spikes_per_channel {
                            let write_idx = (out_offset + count) as usize;
                            out_sample_indices[write_idx] = t;
                            out_amplitudes[write_idx] = val;
                            count = count + 1u32;
                        }
                        has_prev = 1u32;
                        last_spike_t = t;
                    }
                }
                t = t + 1u32;
            }
        }

        out_counts[ch as usize] = count;
    }
}

/// Host-side dispatcher executing [`detect_channel_troughs_kernel`] directly on an in-VRAM
/// filtered trace handle and downloading only the compact spike events.
#[allow(clippy::too_many_arguments)]
pub fn execute_detect_spikes_in_vram<R: Runtime>(
    client: &ComputeClient<R>,
    trace_handle: &cubecl::server::Handle,
    sigmas_handle: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    valid_start: usize,
    valid_end: usize,
    threshold_factor: f32,
    refractory_samples: usize,
    max_spikes_per_channel: usize,
    is_cpu: bool,
) -> Vec<SpikeEvent> {
    if channels == 0 || samples < 3 || valid_start >= valid_end {
        return Vec::new();
    }

    let max_spikes = max_spikes_per_channel.max(1);
    let total_trace_elems = channels * samples;
    let total_spike_slots = channels * max_spikes;

    let out_indices_handle = client.empty(total_spike_slots * std::mem::size_of::<u32>());
    let out_amps_handle = client.empty(total_spike_slots * std::mem::size_of::<f32>());
    let out_counts_handle = client.empty(channels * std::mem::size_of::<u32>());

    let geom = LaunchGeometry::for_channel_sequence(channels, is_cpu);

    unsafe {
        detect_channel_troughs_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(trace_handle.clone(), total_trace_elems),
            ArrayArg::from_raw_parts(sigmas_handle.clone(), channels),
            ArrayArg::from_raw_parts(out_indices_handle.clone(), total_spike_slots),
            ArrayArg::from_raw_parts(out_amps_handle.clone(), total_spike_slots),
            ArrayArg::from_raw_parts(out_counts_handle.clone(), channels),
            channels as u32,
            samples as u32,
            valid_start as u32,
            valid_end as u32,
            threshold_factor,
            refractory_samples as u32,
            max_spikes as u32,
        );
    }

    let counts_bytes = client.read_one(out_counts_handle).expect("VRAM read counts");
    let counts = u32::from_bytes(&counts_bytes);
    let total_detected: usize = counts.iter().map(|&c| c as usize).sum();
    if total_detected == 0 {
        return Vec::new();
    }

    let indices_bytes = client.read_one(out_indices_handle).expect("VRAM read indices");
    let amps_bytes = client.read_one(out_amps_handle).expect("VRAM read amps");
    let indices = u32::from_bytes(&indices_bytes);
    let amps = f32::from_bytes(&amps_bytes);

    let mut events = Vec::with_capacity(total_detected);
    for (ch, &cnt) in counts.iter().enumerate().take(channels) {
        let n = (cnt as usize).min(max_spikes);
        let base = ch * max_spikes;
        for i in 0..n {
            events.push(SpikeEvent {
                channel_id: ch,
                sample_index: indices[base + i] as u64,
                peak_amplitude_uv: amps[base + i],
            });
        }
    }

    events.sort_by_key(|s| s.sample_index);
    events
}
