use serde::{Deserialize, Serialize};

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

    /// Samples of the primary channel (its row of `waveform`; the first row when the primary
    /// channel is not among `channel_ids`).
    pub fn primary_trace(&self) -> &[f32] {
        let row = self.channel_ids.iter().position(|&c| c == self.primary_channel).unwrap_or(0);
        self.waveform.get(row * self.num_samples..(row + 1) * self.num_samples).unwrap_or(&[])
    }
}

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
    /// Sample of the spike peak (the detected sample) within every snippet, i.e. the samples cut
    /// before it.
    pub peak_index: usize,
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
    /// Creates an empty `SnippetBatch` with specified channel and sample dimensions and the peak at
    /// sample `peak_index`.
    pub fn new(num_channels: usize, num_samples: usize, peak_index: usize) -> Self {
        Self {
            data: Vec::new(),
            num_spikes: 0,
            num_channels,
            num_samples,
            peak_index,
            primary_channels: Vec::new(),
            center_samples: Vec::new(),
            subsample_offsets: Vec::new(),
            channel_ids: Vec::new(),
        }
    }

    /// Creates a `SnippetBatch` from a pre-allocated flat tensor buffer `[N, K, T]`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_parts(
        data: Vec<f32>,
        num_spikes: usize,
        num_channels: usize,
        num_samples: usize,
        peak_index: usize,
        primary_channels: Vec<usize>,
        center_samples: Vec<u64>,
        subsample_offsets: Vec<f32>,
        channel_ids: Vec<usize>,
    ) -> Self {
        assert!(peak_index < num_samples.max(1), "peak index outside the snippet");
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
            peak_index,
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
