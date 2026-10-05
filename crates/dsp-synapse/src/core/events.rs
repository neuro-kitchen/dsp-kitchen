use serde::{Deserialize, Serialize};

/// Detected action potential event (spike).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub channel_id: usize,
    pub sample_index: u64,
    pub peak_amplitude_uv: f32,
}

/// Deduplicated multi-channel spike event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeduplicatedSpike {
    /// The primary channel with the deepest negative trough.
    pub primary_channel: usize,
    /// Sample index of the peak on the primary channel.
    pub sample_index: u64,
    /// Peak amplitude in microvolts on the primary channel.
    pub peak_amplitude_uv: f32,
    /// Neighboring channels that also co-detected this action potential.
    pub participating_channels: Vec<usize>,
}

/// Result of template matching / collision deconvolution for a single resolved spike.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedSpike {
    pub unit_id: usize,
    pub sample_index: u64,
    pub subsample_lag: f32,
    pub amplitude_scale: f32,
    pub score: f32,
}
