use serde::{Deserialize, Serialize};
use crate::detection::SpikeEvent;
use super::alignment::parabolic_subsample_offset;

/// Extracted spike waveform snippet across one or multiple local channels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveformSnippet {
    pub channel_id: usize,
    pub center_sample: u64,
    /// Sub-sample offset in samples [-0.5, +0.5] derived via parabolic interpolation.
    pub subsample_offset: f32,
    pub num_channels: usize,
    pub num_samples: usize,
    /// Flat array of shape [num_channels, num_samples].
    pub waveform: Vec<f32>,
}

/// Extracts multi-channel waveform snippets for a list of detected spike events.
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
            channel_id: ch,
            center_sample: spike.sample_index,
            subsample_offset: sub_offset,
            num_channels: 1,
            num_samples: snippet_len,
            waveform: slice.to_vec(),
        });
    }

    snippets
}
