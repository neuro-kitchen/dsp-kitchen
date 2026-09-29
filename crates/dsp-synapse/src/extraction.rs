use serde::{Deserialize, Serialize};
use crate::detection::SpikeEvent;

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

/// Computes the sub-sample peak offset using 3-point parabolic interpolation:
/// `delta_t = (y[t-1] - y[t+1]) / (2 * (y[t-1] - 2*y[t] + y[t+1]))`
pub fn parabolic_subsample_offset(y_prev: f32, y_peak: f32, y_next: f32) -> f32 {
    let denom = 2.0 * (y_prev - 2.0 * y_peak + y_next);
    if denom.abs() < 1e-6 {
        return 0.0;
    }
    ((y_prev - y_next) / denom).clamp(-0.5, 0.5)
}

/// Extracts multi-channel waveform snippets for a list of detected spike events.
///
/// - `data`: Flat continuous array [channels, samples]
/// - `channels`: Total channels in recording
/// - `samples`: Total temporal samples in recording
/// - `spikes`: Detected spike events
/// - `pre_samples`: Samples before peak (e.g. 20 samples / ~0.66 ms)
/// - `post_samples`: Samples after peak (e.g. 40 samples / ~1.33 ms)
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

        // Skip boundary events that cannot fit full window
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parabolic_alignment() {
        // True minimum is slightly to the right of sample 1 (e.g. 1.2)
        let y0 = -80.0;
        let y1 = -100.0;
        let y2 = -90.0;
        let offset = parabolic_subsample_offset(y0, y1, y2);
        assert!(offset > 0.0 && offset < 0.5);
    }
}
