use serde::{Deserialize, Serialize};
use dsp_core::SensorLayout;
use crate::detection::DeduplicatedSpike;
use crate::probe::find_k_nearest_neighbors;
use super::alignment::parabolic_subsample_offset;
use super::resample::resample_sinc_multichannel;
use super::snippet::WaveformSnippet;

/// Contiguous 3D batch of multi-channel waveform snippets with shape `[num_spikes, num_channels, num_samples]`.
///
/// Designed for zero-copy tensor hand-off between `dsp-synapse` and `dsp-synapse-ml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnippetBatch {
    /// Flat contiguous array of shape `[num_spikes, num_channels, num_samples]`.
    pub data: Vec<f32>,
    pub num_spikes: usize,
    pub num_channels: usize,
    pub num_samples: usize,
    /// Primary electrode channel index for each spike (`len == num_spikes`).
    pub primary_channels: Vec<usize>,
    /// Sample timestamp at peak center for each spike (`len == num_spikes`).
    pub center_samples: Vec<u64>,
    /// Sub-sample parabolic shift in `[-0.5, 0.5]` for each spike (`len == num_spikes`).
    pub subsample_offsets: Vec<f32>,
    /// Flattened K-nearest neighbor channel IDs of shape `[num_spikes, num_channels]`.
    pub channel_ids: Vec<usize>,
}

impl SnippetBatch {
    /// Creates an empty `SnippetBatch` with specified channel and sample dimensions.
    pub fn new(num_channels: usize, num_samples: usize) -> Self {
        Self {
            data: Vec::new(),
            num_spikes: 0,
            num_channels,
            num_samples,
            primary_channels: Vec::new(),
            center_samples: Vec::new(),
            subsample_offsets: Vec::new(),
            channel_ids: Vec::new(),
        }
    }

    /// Creates a `SnippetBatch` from a pre-allocated flat tensor buffer `[N, K, T]`.
    pub fn from_raw_parts(
        data: Vec<f32>,
        num_spikes: usize,
        num_channels: usize,
        num_samples: usize,
        primary_channels: Vec<usize>,
        center_samples: Vec<u64>,
        subsample_offsets: Vec<f32>,
        channel_ids: Vec<usize>,
    ) -> Self {
        assert_eq!(data.len(), num_spikes * num_channels * num_samples);
        assert_eq!(primary_channels.len(), num_spikes);
        assert_eq!(center_samples.len(), num_spikes);
        assert_eq!(subsample_offsets.len(), num_spikes);
        assert_eq!(channel_ids.len(), num_spikes * num_channels);
        Self {
            data,
            num_spikes,
            num_channels,
            num_samples,
            primary_channels,
            center_samples,
            subsample_offsets,
            channel_ids,
        }
    }

    /// Shape of the underlying 3D tensor `[num_spikes, num_channels, num_samples]`.
    pub fn shape(&self) -> [usize; 3] {
        [self.num_spikes, self.num_channels, self.num_samples]
    }

    /// Zero-copy slice of the entire `[N, K, T]` contiguous data buffer.
    pub fn as_flat_slice(&self) -> &[f32] {
        &self.data
    }

    /// Zero-copy mutable slice of the entire `[N, K, T]` contiguous data buffer.
    pub fn as_flat_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Zero-copy `[K, T]` slice for spike `spike_idx`.
    pub fn snippet_slice(&self, spike_idx: usize) -> &[f32] {
        let stride = self.num_channels * self.num_samples;
        let start = spike_idx * stride;
        &self.data[start..start + stride]
    }

    /// Zero-copy `[T]` slice for spike `spike_idx` and local neighbor channel `local_ch`.
    pub fn channel_slice(&self, spike_idx: usize, local_ch: usize) -> &[f32] {
        let start = (spike_idx * self.num_channels + local_ch) * self.num_samples;
        &self.data[start..start + self.num_samples]
    }

    /// Zero-copy `[K]` slice of electrode channel IDs for spike `spike_idx`.
    pub fn spike_channel_ids(&self, spike_idx: usize) -> &[usize] {
        let start = spike_idx * self.num_channels;
        &self.channel_ids[start..start + self.num_channels]
    }

    /// Converts a slice of `WaveformSnippet` into a contiguous `SnippetBatch`.
    pub fn from_snippets(snippets: &[WaveformSnippet]) -> Option<Self> {
        if snippets.is_empty() {
            return None;
        }
        let num_channels = snippets[0].num_channels();
        let num_samples = snippets[0].num_samples;
        let stride = num_channels * num_samples;
        let num_spikes = snippets.len();

        let mut data = Vec::with_capacity(num_spikes * stride);
        let mut primary_channels = Vec::with_capacity(num_spikes);
        let mut center_samples = Vec::with_capacity(num_spikes);
        let mut subsample_offsets = Vec::with_capacity(num_spikes);
        let mut channel_ids = Vec::with_capacity(num_spikes * num_channels);

        for s in snippets {
            if s.waveform.len() != stride || s.channel_ids.len() != num_channels {
                continue;
            }
            data.extend_from_slice(&s.waveform);
            primary_channels.push(s.primary_channel);
            center_samples.push(s.center_sample);
            subsample_offsets.push(s.subsample_offset);
            channel_ids.extend_from_slice(&s.channel_ids);
        }

        let actual_spikes = primary_channels.len();
        Some(Self {
            data,
            num_spikes: actual_spikes,
            num_channels,
            num_samples,
            primary_channels,
            center_samples,
            subsample_offsets,
            channel_ids,
        })
    }

    /// Converts this batch back into individual `WaveformSnippet` structs.
    pub fn to_snippets(&self) -> Vec<WaveformSnippet> {
        let mut out = Vec::with_capacity(self.num_spikes);
        for i in 0..self.num_spikes {
            out.push(WaveformSnippet {
                primary_channel: self.primary_channels[i],
                center_sample: self.center_samples[i],
                subsample_offset: self.subsample_offsets[i],
                channel_ids: self.spike_channel_ids(i).to_vec(),
                num_samples: self.num_samples,
                waveform: self.snippet_slice(i).to_vec(),
            });
        }
        out
    }
}

/// Extracts a contiguous `SnippetBatch` directly from multi-channel raw data without intermediate per-spike heap allocations.
pub fn extract_snippet_batch_multichannel(
    data: &[f32],
    _channels: usize,
    samples: usize,
    spikes: &[DeduplicatedSpike],
    layout: &SensorLayout,
    k_neighbors: usize,
    pre_samples: usize,
    post_samples: usize,
    apply_sinc_shift: bool,
) -> SnippetBatch {
    let snippet_len = pre_samples + post_samples;
    let k = k_neighbors.min(layout.total_channels()).max(1);
    let stride = k * snippet_len;

    let mut flat_data = Vec::with_capacity(spikes.len() * stride);
    let mut primary_channels = Vec::with_capacity(spikes.len());
    let mut center_samples = Vec::with_capacity(spikes.len());
    let mut subsample_offsets = Vec::with_capacity(spikes.len());
    let mut channel_ids = Vec::with_capacity(spikes.len() * k);

    let mut raw_scratch = vec![0.0f32; stride];

    for spike in spikes {
        let center = spike.sample_index as usize;
        let primary_ch = spike.primary_channel;

        if center < pre_samples + 1 || center + post_samples + 1 >= samples {
            continue;
        }

        let neighbor_channels = find_k_nearest_neighbors(layout, primary_ch, k);
        if neighbor_channels.len() != k {
            continue;
        }

        let prim_offset = primary_ch * samples;
        let y_prev = data[prim_offset + center - 1];
        let y_peak = data[prim_offset + center];
        let y_next = data[prim_offset + center + 1];
        let sub_offset = parabolic_subsample_offset(y_prev, y_peak, y_next);

        for (row_idx, &ch) in neighbor_channels.iter().enumerate() {
            let ch_offset = ch * samples;
            let start = ch_offset + center - pre_samples;
            let row_dest = row_idx * snippet_len;
            raw_scratch[row_dest..row_dest + snippet_len]
                .copy_from_slice(&data[start..start + snippet_len]);
        }

        if apply_sinc_shift && sub_offset.abs() > 1e-4 {
            let aligned = resample_sinc_multichannel(&raw_scratch, k, snippet_len, -sub_offset, 5);
            flat_data.extend_from_slice(&aligned);
        } else {
            flat_data.extend_from_slice(&raw_scratch);
        }

        primary_channels.push(primary_ch);
        center_samples.push(spike.sample_index);
        subsample_offsets.push(sub_offset);
        channel_ids.extend_from_slice(&neighbor_channels);
    }

    let num_spikes = primary_channels.len();
    SnippetBatch {
        data: flat_data,
        num_spikes,
        num_channels: k,
        num_samples: snippet_len,
        primary_channels,
        center_samples,
        subsample_offsets,
        channel_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::tetrode;

    #[test]
    fn test_snippet_batch_extraction_and_indexing() {
        let layout = tetrode();
        let samples = 500;
        let channels = 4;
        let mut data = vec![0.0f32; channels * samples];

        // Inject synthetic spike at sample 200 on channel 1
        for ch in 0..channels {
            data[ch * samples + 200] = -100.0 / (1.0 + ch as f32);
        }

        let spikes = vec![DeduplicatedSpike {
            primary_channel: 1,
            sample_index: 200,
            peak_amplitude_uv: -50.0,
            participating_channels: vec![0, 1, 2, 3],
        }];

        let batch = extract_snippet_batch_multichannel(
            &data,
            channels,
            samples,
            &spikes,
            &layout,
            4,
            10,
            20,
            false,
        );

        assert_eq!(batch.shape(), [1, 4, 30]);
        assert_eq!(batch.snippet_slice(0).len(), 120);
        assert_eq!(batch.channel_slice(0, 0).len(), 30);

        let roundtrip = SnippetBatch::from_snippets(&batch.to_snippets()).unwrap();
        assert_eq!(roundtrip, batch);
    }
}
