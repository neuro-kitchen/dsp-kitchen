use cubecl::prelude::*;
use dsp_base::geometry::LaunchGeometry;

/// CubeCL kernel reducing extracted `[num_spikes, k_neighbors, snippet_len]` snippets in VRAM
/// into per-channel `[channels, k_neighbors, snippet_len]` first-moment (`sum`) and second-moment
/// (`sum_sq`) tensors, plus `out_counts[channels]`.
///
/// Parallelized with 1 GPU thread per `(channel, k_idx, sample_idx)` slot in
/// `[0, channels * k_neighbors * snippet_len)`.
#[cube(launch)]
pub fn reduce_channel_templates_kernel(
    snippets: &Array<f32>,
    spike_primary_channels: &Array<u32>,
    out_sum: &mut Array<f32>,
    out_sum_sq: &mut Array<f32>,
    out_counts: &mut Array<u32>,
    num_channels: u32,
    num_spikes: u32,
    elems_per_spike: u32,
) {
    let tid = ABSOLUTE_POS_X;
    let total_slots = num_channels * elems_per_spike;

    if tid < total_slots {
        let ch = tid / elems_per_spike;
        let rem = tid - ch * elems_per_spike;

        let mut acc_sum = 0.0f32;
        let mut acc_sum_sq = 0.0f32;
        let mut count = 0u32;

        let mut i = 0u32;
        while i < num_spikes {
            let p_ch = spike_primary_channels[i as usize];
            if p_ch == ch {
                let val = snippets[(i * elems_per_spike + rem) as usize];
                acc_sum = acc_sum + val;
                acc_sum_sq = acc_sum_sq + val * val;
                count = count + 1u32;
            }
            i = i + 1u32;
        }

        out_sum[tid as usize] = acc_sum;
        out_sum_sq[tid as usize] = acc_sum_sq;
        if rem == 0u32 {
            out_counts[ch as usize] = count;
        }
    }
}

/// Host-side batch reduction result downloaded from VRAM (`sum`, `sum_sq`, and `counts` per channel).
pub struct BatchTemplateStats {
    pub counts: Vec<u32>,
    pub sum: Vec<f32>,
    pub sum_sq: Vec<f32>,
}

/// Host-side dispatcher executing [`reduce_channel_templates_kernel`] on the in-VRAM extracted
/// snippets and returning only the compact `[channels * k_neighbors * snippet_len]` batch statistics.
pub fn execute_reduce_templates_in_vram<R: Runtime>(
    client: &ComputeClient<R>,
    snippets_handle: &cubecl::server::Handle,
    prim_channels_handle: &cubecl::server::Handle,
    channels: usize,
    num_spikes: usize,
    k_neighbors: usize,
    snippet_len: usize,
    is_cpu: bool,
) -> BatchTemplateStats {
    let elems_per_spike = k_neighbors * snippet_len;
    let total_slots = channels * elems_per_spike;

    let out_sum_handle = client.empty(total_slots * std::mem::size_of::<f32>());
    let out_sum_sq_handle = client.empty(total_slots * std::mem::size_of::<f32>());
    let out_counts_handle = client.empty(channels * std::mem::size_of::<u32>());

    let geom = LaunchGeometry::for_1d(total_slots, is_cpu);

    unsafe {
        reduce_channel_templates_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(snippets_handle.clone(), num_spikes * elems_per_spike),
            ArrayArg::from_raw_parts(prim_channels_handle.clone(), num_spikes),
            ArrayArg::from_raw_parts(out_sum_handle.clone(), total_slots),
            ArrayArg::from_raw_parts(out_sum_sq_handle.clone(), total_slots),
            ArrayArg::from_raw_parts(out_counts_handle.clone(), channels),
            channels as u32,
            num_spikes as u32,
            elems_per_spike as u32,
        );
    }

    let counts_bytes = client.read_one(out_counts_handle).expect("VRAM read counts");
    let sum_bytes = client.read_one(out_sum_handle).expect("VRAM read sum");
    let sum_sq_bytes = client.read_one(out_sum_sq_handle).expect("VRAM read sum_sq");

    BatchTemplateStats {
        counts: u32::from_bytes(&counts_bytes).to_vec(),
        sum: f32::from_bytes(&sum_bytes).to_vec(),
        sum_sq: f32::from_bytes(&sum_sq_bytes).to_vec(),
    }
}
