use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;

/// CubeCL kernel reducing extracted `[num_spikes, k_neighbors, snippet_len]` snippets in VRAM into
/// per-channel `[channels, k_neighbors, snippet_len]` mean and second central moment ($M_2$).
///
/// Spikes arrive segmented by primary channel: `spike_order[segment_offsets[ch]..segment_offsets[ch + 1]]`
/// lists the spikes of channel `ch`. One unit per `(channel, k_idx, sample_idx)` slot walks only its
/// channel's segment (total work O(N·K·L)) with Welford updates, so no large sums cancel in f32.
#[cube(launch)]
pub fn reduce_channel_templates_kernel(
    snippets: &[f32],
    spike_order: &[u32],
    segment_offsets: &[u32],
    out_mean: &mut [f32],
    out_m2: &mut [f32],
    num_channels: u32,
    elems_per_spike: u32,
) {
    let tid = ABSOLUTE_POS as u32;
    if tid < num_channels * elems_per_spike {
        let ch = tid / elems_per_spike;
        let rem = tid - ch * elems_per_spike;
        let start = segment_offsets[ch as usize];
        let end = segment_offsets[(ch + 1u32) as usize];

        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        let mut j = start;
        while j < end {
            let spike = spike_order[j as usize];
            let val = snippets[(spike * elems_per_spike + rem) as usize];
            let n = (j - start + 1u32) as f32;
            let delta = val - mean;
            mean += delta / n;
            m2 += delta * (val - mean);
            j += 1u32;
        }

        out_mean[tid as usize] = mean;
        out_m2[tid as usize] = m2;
    }
}

/// Per-channel batch moments downloaded from VRAM: `counts[channels]`, and `mean` / `m2` of
/// shape `[channels, k_neighbors, snippet_len]`.
pub struct BatchTemplateStats {
    pub counts: Vec<u32>,
    pub mean: Vec<f32>,
    pub m2: Vec<f32>,
}

/// Host-side dispatcher executing [`reduce_channel_templates_kernel`] on in-VRAM extracted snippets.
/// `primary_channels[i]` is the primary channel of snippet `i`; snippets on channels
/// `>= channels` are ignored.
pub fn execute_reduce_templates_in_vram(
    client: &Client,
    snippets_handle: &cubecl::server::Handle,
    primary_channels: &[u32],
    channels: usize,
    k_neighbors: usize,
    snippet_len: usize,
) -> BatchTemplateStats {
    let num_spikes = primary_channels.len();
    let elems_per_spike = k_neighbors * snippet_len;
    let total_slots = channels * elems_per_spike;

    // Counting sort of the snippets by primary channel.
    let mut counts = vec![0u32; channels];
    for &ch in primary_channels {
        if let Some(c) = counts.get_mut(ch as usize) {
            *c += 1;
        }
    }
    let mut offsets = vec![0u32; channels + 1];
    for ch in 0..channels {
        offsets[ch + 1] = offsets[ch] + counts[ch];
    }
    let mut cursor = offsets.clone();
    let mut order = vec![0u32; offsets[channels] as usize];
    for (i, &ch) in primary_channels.iter().enumerate() {
        let ch = ch as usize;
        if ch < channels {
            order[cursor[ch] as usize] = i as u32;
            cursor[ch] += 1;
        }
    }
    if order.is_empty() || total_slots == 0 {
        return BatchTemplateStats { counts, mean: vec![0.0; total_slots], m2: vec![0.0; total_slots] };
    }

    let order_handle = client.create_from_slice(u32::as_bytes(&order));
    let offsets_handle = client.create_from_slice(u32::as_bytes(&offsets));
    let out_mean_handle = client.empty(total_slots * std::mem::size_of::<f32>());
    let out_m2_handle = client.empty(total_slots * std::mem::size_of::<f32>());

    let geom = LaunchGeometry::elementwise(client, total_slots);
    unsafe {
        reduce_channel_templates_kernel::launch(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(snippets_handle.clone(), num_spikes * elems_per_spike),
            BufferArg::from_raw_parts(order_handle, order.len()),
            BufferArg::from_raw_parts(offsets_handle, channels + 1),
            BufferArg::from_raw_parts(out_mean_handle.clone(), total_slots),
            BufferArg::from_raw_parts(out_m2_handle.clone(), total_slots),
            channels as u32,
            elems_per_spike as u32,
        );
    }

    let mean_bytes = client.read_one(out_mean_handle).expect("VRAM read mean");
    let m2_bytes = client.read_one(out_m2_handle).expect("VRAM read m2");
    BatchTemplateStats {
        counts,
        mean: f32::from_bytes(&mean_bytes).to_vec(),
        m2: f32::from_bytes(&m2_bytes).to_vec(),
    }
}
