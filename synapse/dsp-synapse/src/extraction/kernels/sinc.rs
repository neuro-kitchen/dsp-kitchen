//! Snippet extraction with sub-sample realignment in device memory: one unit per spike finds its
//! trough shift, dsp-base turns the shifts into windowed-sinc taps once per spike
//! (`dsp_base::resampler::fractional_delay_taps_device`), one unit per output sample applies them.

use cubecl::prelude::*;
use dsp_base::core::{buffer, DspFloat};
use dsp_base::resampler::fractional_delay_taps_device;
use dsp_core::compute::LaunchGeometry;

use crate::detection::DeduplicatedSpike;
use crate::extraction::{SINC_KERNEL_RADIUS, snippet_fits};

/// Denominator below which the three samples around a trough count as flat (no shift).
const PARABOLA_FLAT: f32 = 1e-12;

/// One unit per spike: the parabolic sub-sample offset `δ ∈ [−½, ½]` of the trough at
/// `center_samples[i]` on `primary_channels[i]` (trough at `center + δ`).
#[cube(launch)]
pub fn trough_shift_kernel<F: Float>(
    trace: &[F],
    primary_channels: &[u32],
    center_samples: &[u32],
    shifts: &mut [F],
    num_samples: u32,
    num_spikes: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < num_spikes {
        let base = primary_channels[i as usize] * num_samples + center_samples[i as usize];
        let y_prev = trace[(base - 1u32) as usize];
        let y_mid = trace[base as usize];
        let y_next = trace[(base + 1u32) as usize];
        let denom = F::new(2.0f32) * (y_prev - F::new(2.0f32) * y_mid + y_next);
        let mut shift = F::new(0.0f32);
        if F::abs(denom) > F::new(PARABOLA_FLAT) {
            shift = F::clamp((y_prev - y_next) / denom, F::new(-0.5f32), F::new(0.5f32));
        }
        shifts[i as usize] = shift;
    }
}

/// One unit per output sample of `[num_spikes, k_neighbors, snippet_len]`: sample
/// `center − pre + s` of the `k`-th neighbour of the spike's primary channel, as
/// `Σₘ taps[m] · x[· + m − radius]` when `apply_shift` (so the trough lands on sample `pre`), else
/// copied. Spikes were checked to fit (with the taps) beforehand.
#[cube(launch)]
pub fn extract_snippets_kernel<F: Float>(
    trace: &[F],
    knn_table: &[u32],
    primary_channels: &[u32],
    center_samples: &[u32],
    taps: &[F],
    out_snippets: &mut [F],
    num_samples: u32,
    num_spikes: u32,
    k_neighbors: u32,
    pre_samples: u32,
    snippet_len: u32,
    apply_shift: u32,
    #[comptime] radius: u32,
) {
    let tid = ABSOLUTE_POS as u32;
    let per_spike = k_neighbors * snippet_len;
    if tid < num_spikes * per_spike {
        let spike = tid / per_spike;
        let rem = tid - spike * per_spike;
        let k = rem / snippet_len;
        let s = rem - k * snippet_len;
        let primary = primary_channels[spike as usize];
        let neighbour = knn_table[(primary * k_neighbors + k) as usize];
        let pos = neighbour * num_samples + center_samples[spike as usize] - pre_samples + s;
        if apply_shift == 0u32 {
            out_snippets[tid as usize] = trace[pos as usize];
        } else {
            let n = 2 * radius + 1;
            let tap_base = spike * n;
            let mut sum = F::new(0.0f32);
            #[unroll]
            for m in 0..n {
                sum += taps[(tap_base + m) as usize] * trace[(pos + m - radius) as usize];
            }
            out_snippets[tid as usize] = sum;
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

/// Extracts realigned snippets of `spikes` from the in-VRAM filtered trace (`[channels, samples]`
/// of `F`) into a `[spikes, k_neighbors, pre + post]` device tensor.
///
/// Spikes whose window plus [`crate::extraction::extraction_margin`] leaves the trace are skipped
/// (reported in [`VramSnippets::dropped`]). Returns `None` when nothing is extracted.
#[allow(clippy::too_many_arguments)]
pub fn execute_extract_sinc_in_vram<F: DspFloat>(
    client: &Client,
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
    let prim_handle = buffer::upload(client, &primary_channels);
    let center_handle = buffer::upload(client, &center_samples);
    let total = num_spikes * k_neighbors * snippet_len;
    let out_snippets_handle = buffer::empty::<F>(client, total);

    // SAFETY (all launches): handles hold the lengths passed — the trace `channels · samples`,
    // the KNN table `channels · k_neighbors`, per-spike arrays `num_spikes`, taps
    // `num_spikes · (2·radius + 1)`, the output `total`
    let taps = if apply_sinc_shift {
        let shifts = buffer::empty::<F>(client, num_spikes);
        let per_spike = LaunchGeometry::elementwise(client, num_spikes);
        unsafe {
            trough_shift_kernel::launch::<F>(
                client,
                per_spike.cube_count,
                per_spike.cube_dim,
                BufferArg::from_raw_parts(trace_handle.clone(), channels * samples),
                BufferArg::from_raw_parts(prim_handle.clone(), num_spikes),
                BufferArg::from_raw_parts(center_handle.clone(), num_spikes),
                BufferArg::from_raw_parts(shifts.clone(), num_spikes),
                samples as u32,
                num_spikes as u32,
            );
        }
        fractional_delay_taps_device::<F>(client, &shifts, num_spikes, SINC_KERNEL_RADIUS)
    } else {
        buffer::empty::<F>(client, 1)
    };
    let tap_len = if apply_sinc_shift { num_spikes * (2 * SINC_KERNEL_RADIUS + 1) } else { 1 };

    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        extract_snippets_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(trace_handle.clone(), channels * samples),
            BufferArg::from_raw_parts(knn_handle.clone(), channels * k_neighbors),
            BufferArg::from_raw_parts(prim_handle.clone(), num_spikes),
            BufferArg::from_raw_parts(center_handle, num_spikes),
            BufferArg::from_raw_parts(taps, tap_len),
            BufferArg::from_raw_parts(out_snippets_handle.clone(), total),
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
