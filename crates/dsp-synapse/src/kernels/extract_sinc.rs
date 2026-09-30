use cubecl::prelude::*;
use dsp_base::geometry::LaunchGeometry;
use crate::detection::DeduplicatedSpike;

/// CubeCL kernel for multi-channel waveform snippet extraction and Blackman-Harris windowed sinc
/// fractional realignment directly inside VRAM.
///
/// Parallelized with 1 thread per output sample `ABSOLUTE_POS_X` in `[0, num_spikes * k_neighbors * snippet_len)`.
/// Each thread:
/// 1. Resolves `(spike_idx, k_idx, sample_idx)`.
/// 2. Looks up `primary_channel` and `center_sample` for `spike_idx`.
/// 3. Computes the 3-point parabolic sub-sample offset $\Delta t \in [-0.5, +0.5]$ on `primary_channel`.
/// 4. Looks up the $k$-th neighbor channel from `knn_table[primary_channel * k_neighbors + k_idx]`.
/// 5. Evaluates the Blackman-Harris windowed sinc interpolation around `center - pre_samples + sample_idx`.
#[cube(launch)]
pub fn extract_sinc_snippets_kernel(
    trace: &Array<f32>,
    knn_table: &Array<u32>,
    spike_primary_channels: &Array<u32>,
    spike_center_samples: &Array<u32>,
    out_snippets: &mut Array<f32>,
    num_samples: u32,
    num_spikes: u32,
    k_neighbors: u32,
    pre_samples: u32,
    snippet_len: u32,
    apply_sinc_shift: u32,
) {
    let tid = ABSOLUTE_POS_X;
    let elems_per_spike = k_neighbors * snippet_len;
    let total_elems = num_spikes * elems_per_spike;

    if tid < total_elems {
        let spike_idx = tid / elems_per_spike;
        let rem = tid - spike_idx * elems_per_spike;
        let k_idx = rem / snippet_len;
        let s_idx = rem - k_idx * snippet_len;

        let primary_ch = spike_primary_channels[spike_idx as usize];
        let center = spike_center_samples[spike_idx as usize];
        let nbr_ch = knn_table[(primary_ch * k_neighbors + k_idx) as usize];

        let prim_base = primary_ch * num_samples;
        let y_prev = trace[(prim_base + center - 1u32) as usize];
        let y_peak = trace[(prim_base + center) as usize];
        let y_next = trace[(prim_base + center + 1u32) as usize];

        let denom = 2.0f32 * (y_prev - 2.0f32 * y_peak + y_next);
        let mut sub_offset = 0.0f32;
        if f32::abs(denom) > 1e-6f32 {
            sub_offset = f32::clamp((y_prev - y_next) / denom, -0.5f32, 0.5f32);
        }

        let shift_samples = -sub_offset;
        let ch_base = nbr_ch * num_samples;
        let snippet_start = center - pre_samples;

        if apply_sinc_shift == 0u32 || f32::abs(shift_samples) <= 1e-4f32 {
            out_snippets[tid as usize] = trace[(ch_base + snippet_start + s_idx) as usize];
        } else {
            let pi = 3.14159265f32;
            let radius = 5u32;
            let half_w = 5.0f32;
            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            let mut m = 0u32;
            let diameter = 2u32 * radius + 1u32;

            while m < diameter {
                let target_s_plus_r = s_idx + m;
                if target_s_plus_r >= radius {
                    let src_s = target_s_plus_r - radius;
                    if src_s < snippet_len {
                        let k_f32 = f32::cast_from(m) - half_w;
                        let tau = k_f32 - shift_samples;

                        let norm = f32::clamp(tau / half_w, -1.0f32, 1.0f32);
                        let u = (norm + 1.0f32) * 0.5f32;
                        let two_pi_u = 2.0f32 * pi * u;
                        let w = 0.35875f32
                            - 0.48829f32 * f32::cos(two_pi_u)
                            + 0.14128f32 * f32::cos(2.0f32 * two_pi_u)
                            - 0.01168f32 * f32::cos(3.0f32 * two_pi_u);

                        let abs_tau = f32::abs(tau);
                        let mut sinc_val = 1.0f32;
                        if abs_tau >= 1e-6f32 {
                            let pix = pi * tau;
                            sinc_val = f32::sin(pix) / pix;
                        }

                        let weight = sinc_val * w;
                        let sample_val = trace[(ch_base + snippet_start + src_s) as usize];
                        sum = sum + sample_val * weight;
                        weight_sum = weight_sum + weight;
                    }
                }
                m = m + 1u32;
            }

            if f32::abs(weight_sum) > 1e-6f32 {
                out_snippets[tid as usize] = sum / weight_sum;
            } else {
                out_snippets[tid as usize] = trace[(ch_base + snippet_start + s_idx) as usize];
            }
        }
    }
}

/// Host-side dispatcher executing [`extract_sinc_snippets_kernel`] on the in-VRAM filtered trace
/// and returning a VRAM handle to the extracted `[num_spikes, k_neighbors, snippet_len]` tensor
/// along with the uploaded `spike_primary_channels` handle for downstream template reduction.
#[allow(clippy::too_many_arguments)]
pub fn execute_extract_sinc_in_vram<R: Runtime>(
    client: &ComputeClient<R>,
    trace_handle: &cubecl::server::Handle,
    knn_handle: &cubecl::server::Handle,
    channels: usize,
    samples: usize,
    spikes: &[DeduplicatedSpike],
    k_neighbors: usize,
    pre_samples: usize,
    post_samples: usize,
    apply_sinc_shift: bool,
    is_cpu: bool,
) -> Option<(cubecl::server::Handle, cubecl::server::Handle)> {
    let num_spikes = spikes.len();
    let snippet_len = pre_samples + post_samples;
    if num_spikes == 0 || snippet_len == 0 {
        return None;
    }

    let mut primary_channels = Vec::with_capacity(num_spikes);
    let mut center_samples = Vec::with_capacity(num_spikes);
    for s in spikes {
        primary_channels.push(s.primary_channel as u32);
        center_samples.push(s.sample_index as u32);
    }

    let prim_handle = client.create_from_slice(u32::as_bytes(&primary_channels));
    let center_handle = client.create_from_slice(u32::as_bytes(&center_samples));

    let total_out_elems = num_spikes * k_neighbors * snippet_len;
    let out_snippets_handle = client.empty(total_out_elems * std::mem::size_of::<f32>());

    let geom = LaunchGeometry::for_1d(total_out_elems, is_cpu);

    unsafe {
        extract_sinc_snippets_kernel::launch::<R>(
            client,
            geom.cube_count,
            geom.cube_dim,
            ArrayArg::from_raw_parts(trace_handle.clone(), channels * samples),
            ArrayArg::from_raw_parts(knn_handle.clone(), channels * k_neighbors),
            ArrayArg::from_raw_parts(prim_handle.clone(), num_spikes),
            ArrayArg::from_raw_parts(center_handle, num_spikes),
            ArrayArg::from_raw_parts(out_snippets_handle.clone(), total_out_elems),
            samples as u32,
            num_spikes as u32,
            k_neighbors as u32,
            pre_samples as u32,
            snippet_len as u32,
            u32::from(apply_sinc_shift),
        );
    }

    Some((out_snippets_handle, prim_handle))
}
