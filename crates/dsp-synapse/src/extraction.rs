use serde::{Deserialize, Serialize};

/// Fixed-length extracted spike waveform snippet (e.g. 60 samples / 2 ms @ 30 kHz).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveformSnippet {
    pub channel_id: usize,
    pub center_sample: u64,
    pub samples: Vec<f32>,
}

/// Principal Component Analysis projection features for spike sorting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcaFeatures {
    pub pc1: f32,
    pub pc2: f32,
    pub pc3: f32,
}
