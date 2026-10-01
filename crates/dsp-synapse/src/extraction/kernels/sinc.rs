use cubecl::prelude::*;
use dsp_core::compute::LaunchGeometry;
use crate::detection::DeduplicatedSpike;
use crate::extraction::{SINC_KERNEL_RADIUS, snippet_fits};

/// CubeCL kernel for multi-channel waveform snippet extraction and Blackman-Harris windowed sinc
/// fractional realignment directly inside VRAM.
///
/// Parallelized with 1 thread per output sample `ABSOLUTE_POS` in `[0, num_spikes * k_neighbors * snippet_len)`.
/// Each thread:
/// 1. Resolves `(spike_idx, k_idx, sample_idx)`.
/// 2. Looks up `primary_channel` and `center_sample` for `spike_idx`.
/// 3. Computes the 3-point parabolic sub-sample trough offset $\delta \in [-0.5, +0.5]$ on `primary_channel`
///    (trough at `center + δ`).
/// 4. Looks up the $k$-th neighbor channel from `knn_table[primary_channel * k_neighbors + k_idx]`.
/// 5. Evaluates `x(center − pre + sample_idx + δ)` with the full `2·radius + 1` windowed-sinc taps
///    read from the trace, so the trough lands on sample `pre` (same formula as
///    [`crate::extraction::interpolate_window`]).
///
/// Spikes whose window plus `margin` leaves the trace (or with an invalid channel) produce zeros;
/// the host dispatcher filters them out beforehand.
#[cube(launch)]
pub fn extract_sinc_snippets_kernel(
    trace: &Array<f32>,
    knn_table: &Array<u32>,
    spike_primary_channels: &Array<u32>,
    spike_center_samples: &Array<u32>,
    out_snippets: &mut Array<f32>,
    num_channels: u32,
    num_samples: u32,
    num_spikes: u32,
    k_neighbors: u32,
    pre_samples: u32,
    snippet_len: u32,
    apply_sinc_shift: u32,
    #[comptime] radius: u32,
) {
    let tid = ABSOLUTE_POS as u32;
    let elems_per_spike = k_neighbors * snippet_len;
    let total_elems = num_spikes * elems_per_spike;

    if tid < total_elems {
        let spike_idx = tid / elems_per_spike;
        let rem = tid - spike_idx * elems_per_spike;
        let k_idx = rem / snippet_len;
        let s_idx = rem - k_idx * snippet_len;

        let primary_ch = spike_primary_channels[spike_idx as usize];
        let center = spike_center_samples[spike_idx as usize];
        let post_samples = snippet_len - pre_samples;
        let mut margin = 1u32;
        if apply_sinc_shift != 0u32 {
            margin = radius;
        }

        let valid = primary_ch < num_channels
            && center >= pre_samples + margin
            && center + post_samples + margin <= num_samples;

        if !valid {
            out_snippets[tid as usize] = 0.0f32;
        } else {
            let nbr_ch = knn_table[(primary_ch * k_neighbors + k_idx) as usize];
            let prim_base = primary_ch * num_samples;
            let y_prev = trace[(prim_base + center - 1u32) as usize];
            let y_peak = trace[(prim_base + center) as usize];
            let y_next = trace[(prim_base + center + 1u32) as usize];

            let denom = 2.0f32 * (y_prev - 2.0f32 * y_peak + y_next);
            let mut shift = 0.0f32;
            if f32::abs(denom) > 1e-6f32 {
                shift = f32::clamp((y_prev - y_next) / denom, -0.5f32, 0.5f32);
            }

            let ch_base = nbr_ch * num_samples;
            let pos = ch_base + center - pre_samples + s_idx;

            if apply_sinc_shift == 0u32 || f32::abs(shift) <= 1e-4f32 {
                out_snippets[tid as usize] = trace[pos as usize];
            } else {
                let pi = core::f32::consts::PI;
                let half_w = f32::cast_from(radius);
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                #[unroll]
                for m in 0..2 * radius + 1 {
                    let tau = f32::cast_from(m) - half_w - shift;

                    let norm = f32::clamp(tau / half_w, -1.0f32, 1.0f32);
                    let u = (norm + 1.0f32) * 0.5f32;
                    let two_pi_u = 2.0f32 * pi * u;
                    let w = 0.35875f32
                        - 0.48829f32 * f32::cos(two_pi_u)
                        + 0.14128f32 * f32::cos(2.0f32 * two_pi_u)
                        - 0.01168f32 * f32::cos(3.0f32 * two_pi_u);

                    let mut sinc_val = 1.0f32;
                    if f32::abs(tau) >= 1e-7f32 {
                        let pix = pi * tau;
                        sinc_val = f32::sin(pix) / pix;
                    }

                    let weight = sinc_val * w;
                    sum += trace[(pos + m - radius) as usize] * weight;
                    weight_sum += weight;
                }

                out_snippets[tid as usize] = sum / weight_sum;
            }
        }
    }
}

/// Snippets extracted in VRAM by [`execute_extract_sinc_in_vram`].
#[derive(Debug, Clone)]
pub struct VramSnippets {
    /// `[num_spikes, k_neighbors, snippet_len]` snippet tensor.
    pub snippets: cubecl::server::Handle,
    /// Primary channel per extracted spike (`u32`, `len == num_spikes`).
    pub primary_channels: cubecl::server::Handle,
    /// The same primary channels on the host.
    pub primaries: Vec<u32>,
    /// Indices into the input `spikes` of the extracted spikes, in tensor order.
    pub kept: Vec<usize>,
    /// Spikes skipped because their window leaves the trace or their channel is invalid.
    pub dropped: usize,
}

impl VramSnippets {
    pub fn num_spikes(&self) -> usize {
        self.kept.len()
    }
}

/// Host-side dispatcher executing [`extract_sinc_snippets_kernel`] on the in-VRAM filtered trace.
///
/// Spikes whose window plus [`crate::extraction::extraction_margin`] leaves the trace are skipped
/// (reported in [`VramSnippets::dropped`]). Returns `None` when nothing is extracted.
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
) -> Option<VramSnippets> {
    let snippet_len = pre_samples + post_samples;
    if snippet_len == 0 {
        return None;
    }

    let kept: Vec<usize> = spikes
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            s.primary_channel < channels
                && snippet_fits(s.sample_index as usize, pre_samples, post_samples, samples, apply_sinc_shift)
        })
        .map(|(i, _)| i)
        .collect();
    let dropped = spikes.len() - kept.len();
    let num_spikes = kept.len();
    if num_spikes == 0 {
        return None;
    }

    let primary_channels: Vec<u32> = kept.iter().map(|&i| spikes[i].primary_channel as u32).collect();
    let center_samples: Vec<u32> = kept.iter().map(|&i| spikes[i].sample_index as u32).collect();

    let prim_handle = client.create_from_slice(u32::as_bytes(&primary_channels));
    let center_handle = client.create_from_slice(u32::as_bytes(&center_samples));

    let total_out_elems = num_spikes * k_neighbors * snippet_len;
    let out_snippets_handle = client.empty(total_out_elems * std::mem::size_of::<f32>());

    let geom = LaunchGeometry::elementwise(client, total_out_elems);

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
            channels as u32,
            samples as u32,
            num_spikes as u32,
            k_neighbors as u32,
            pre_samples as u32,
            snippet_len as u32,
            u32::from(apply_sinc_shift),
            SINC_KERNEL_RADIUS as u32,
        );
    }

    Some(VramSnippets { snippets: out_snippets_handle, primary_channels: prim_handle, primaries: primary_channels, kept, dropped })
}
