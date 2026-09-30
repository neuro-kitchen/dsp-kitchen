use serde::{Deserialize, Serialize};
use dsp_core::SensorLayout;
use crate::detection::{SpikeEvent, DeduplicatedSpike};
use crate::probe::find_k_nearest_neighbors;
use super::alignment::parabolic_subsample_offset;
use super::resample::{SINC_KERNEL_RADIUS, interpolate_window};

/// Samples a snippet `[center − pre, center + post)` needs beyond its window on each side: the
/// sinc taps when realigning, else the ±1 neighbours of the parabolic trough fit.
pub fn extraction_margin(apply_sinc_shift: bool) -> usize {
    if apply_sinc_shift { SINC_KERNEL_RADIUS.max(1) } else { 1 }
}

/// Whether a snippet around `center` (plus [`extraction_margin`]) lies inside `0..samples`.
/// Extractors skip spikes that do not fit.
pub fn snippet_fits(center: usize, pre: usize, post: usize, samples: usize, apply_sinc_shift: bool) -> bool {
    let m = extraction_margin(apply_sinc_shift);
    center >= pre + m && center + post + m <= samples
}

/// Parabolic sub-sample trough offset `δ` on `row` at `center` (trough at `center + δ`).
pub(crate) fn trough_offset(row: &[f32], center: usize) -> f32 {
    parabolic_subsample_offset(row[center - 1], row[center], row[center + 1])
}

/// Copies `[center − pre, center − pre + out.len())` of `row` into `out`, realigned by `+δ` so the
/// trough at `center + δ` lands on `out[pre]` when `shift` is given.
pub(crate) fn cut_row(row: &[f32], center: usize, pre: usize, shift: Option<f32>, out: &mut [f32]) {
    let start = center - pre;
    match shift {
        Some(delta) if delta.abs() > 1e-4 => interpolate_window(row, start, delta, SINC_KERNEL_RADIUS, out),
        _ => out.copy_from_slice(&row[start..start + out.len()]),
    }
}

/// Extracted spike waveform snippet across one or multiple local channels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveformSnippet {
    pub primary_channel: usize,
    pub center_sample: u64,
    /// Sub-sample offset in samples [-0.5, +0.5] derived via parabolic interpolation.
    pub subsample_offset: f32,
    /// Channel IDs corresponding to rows of the 2D waveform.
    pub channel_ids: Vec<usize>,
    pub num_samples: usize,
    /// Flat array of shape [num_channels, num_samples].
    pub waveform: Vec<f32>,
}

impl WaveformSnippet {
    pub fn num_channels(&self) -> usize {
        self.channel_ids.len()
    }
}

/// Extracts multi-channel waveform snippets across K-nearest neighbors for deduplicated spike events.
///
/// With `apply_sinc_shift`, each snippet is realigned by windowed-sinc interpolation so the primary
/// channel's sub-sample trough lands on sample `pre_samples`. Spikes whose window (plus
/// [`extraction_margin`]) leaves the buffer are skipped; compare `center_sample`s to find them.
pub fn extract_snippets_multichannel(
    data: &[f32],
    _channels: usize,
    samples: usize,
    spikes: &[DeduplicatedSpike],
    layout: &SensorLayout,
    k_neighbors: usize,
    pre_samples: usize,
    post_samples: usize,
    apply_sinc_shift: bool,
) -> Vec<WaveformSnippet> {
    let snippet_len = pre_samples + post_samples;
    let total_ch = layout.total_channels();
    let knn_per_ch: Vec<Vec<usize>> = (0..total_ch)
        .map(|ch| find_k_nearest_neighbors(layout, ch, k_neighbors))
        .collect();
    let mut snippets = Vec::with_capacity(spikes.len());

    for spike in spikes {
        let center = spike.sample_index as usize;
        let primary_ch = spike.primary_channel;

        if !snippet_fits(center, pre_samples, post_samples, samples, apply_sinc_shift) {
            continue;
        }

        // 1. Lookup precomputed K-nearest neighbor channels around primary channel
        let neighbor_channels = knn_per_ch
            .get(primary_ch)
            .cloned()
            .unwrap_or_else(|| find_k_nearest_neighbors(layout, primary_ch, k_neighbors));
        let k = neighbor_channels.len();

        // 2. Sub-sample trough offset on the primary channel
        let sub_offset = trough_offset(&data[primary_ch * samples..(primary_ch + 1) * samples], center);
        let shift = apply_sinc_shift.then_some(sub_offset);

        // 3. Cut (and realign) each neighbour row
        let mut final_waveform = vec![0.0f32; k * snippet_len];
        for (row_idx, &ch) in neighbor_channels.iter().enumerate() {
            let row = &data[ch * samples..(ch + 1) * samples];
            cut_row(row, center, pre_samples, shift, &mut final_waveform[row_idx * snippet_len..(row_idx + 1) * snippet_len]);
        }

        snippets.push(WaveformSnippet {
            primary_channel: primary_ch,
            center_sample: spike.sample_index,
            subsample_offset: sub_offset,
            channel_ids: neighbor_channels,
            num_samples: snippet_len,
            waveform: final_waveform,
        });
    }

    snippets
}

/// Legacy single-channel snippet extraction wrapper.
pub fn extract_snippets_single_channel(
    data: &[f32],
    _channels: usize,
    samples: usize,
    spikes: &[SpikeEvent],
    pre_samples: usize,
    post_samples: usize,
) -> Vec<WaveformSnippet> {
    let snippet_len = pre_samples + post_samples;
    let mut snippets = Vec::with_capacity(spikes.len());

    for spike in spikes {
        let center = spike.sample_index as usize;
        let ch = spike.channel_id;

        if !snippet_fits(center, pre_samples, post_samples, samples, false) {
            continue;
        }

        let row = &data[ch * samples..(ch + 1) * samples];
        let slice = &row[center - pre_samples..center - pre_samples + snippet_len];
        let sub_offset = trough_offset(row, center);

        snippets.push(WaveformSnippet {
            primary_channel: ch,
            center_sample: spike.sample_index,
            subsample_offset: sub_offset,
            channel_ids: vec![ch],
            num_samples: snippet_len,
            waveform: slice.to_vec(),
        });
    }

    snippets
}
