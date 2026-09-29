use serde::{Deserialize, Serialize};
use dsp_core::SensorLayout;
use crate::detection::{SpikeEvent, DeduplicatedSpike};
use crate::probe::find_k_nearest_neighbors;
use super::alignment::parabolic_subsample_offset;
use super::resample::resample_sinc_multichannel;

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
    let mut snippets = Vec::with_capacity(spikes.len());

    for spike in spikes {
        let center = spike.sample_index as usize;
        let primary_ch = spike.primary_channel;

        if center < pre_samples + 1 || center + post_samples + 1 >= samples {
            continue;
        }

        // 1. Identify K-nearest neighbor channels around primary channel
        let neighbor_channels = find_k_nearest_neighbors(layout, primary_ch, k_neighbors);
        let k = neighbor_channels.len();

        // 2. Compute parabolic sub-sample offset on primary channel
        let prim_offset = primary_ch * samples;
        let y_prev = data[prim_offset + center - 1];
        let y_peak = data[prim_offset + center];
        let y_next = data[prim_offset + center + 1];
        let sub_offset = parabolic_subsample_offset(y_prev, y_peak, y_next);

        // 3. Cut multi-channel slice [K, snippet_len]
        let mut raw_snippet = vec![0.0f32; k * snippet_len];
        for (row_idx, &ch) in neighbor_channels.iter().enumerate() {
            let ch_offset = ch * samples;
            let start = ch_offset + center - pre_samples;
            let row_dest = row_idx * snippet_len;
            raw_snippet[row_dest..row_dest + snippet_len].copy_from_slice(&data[start..start + snippet_len]);
        }

        // 4. Optionally apply continuous sinc realignment by -sub_offset
        let final_waveform = if apply_sinc_shift && sub_offset.abs() > 1e-4 {
            resample_sinc_multichannel(&raw_snippet, k, snippet_len, -sub_offset, 5)
        } else {
            raw_snippet
        };

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

        if center < pre_samples + 1 || center + post_samples + 1 >= samples {
            continue;
        }

        let ch_offset = ch * samples;
        let start = ch_offset + center - pre_samples;
        let slice = &data[start..start + snippet_len];

        let y_prev = data[ch_offset + center - 1];
        let y_peak = data[ch_offset + center];
        let y_next = data[ch_offset + center + 1];
        let sub_offset = parabolic_subsample_offset(y_prev, y_peak, y_next);

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
