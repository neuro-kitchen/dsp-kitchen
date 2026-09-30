use serde::{Deserialize, Serialize};

/// Triggered waveform snippets (e.g. spikes or evoked responses) with per-snippet metadata.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SnippetSeries {
    pub name: String,
    pub description: String,
    pub sample_rate: f64,
    pub samples_per_snippet: usize,
    /// Seconds from the session start.
    pub timestamps: Vec<f64>,
    pub channels: Vec<u16>,
    /// Sort codes / unit labels (0 = unsorted).
    pub sort_codes: Vec<u16>,
    /// Snippets back to back: `samples_per_snippet` values each, scaled to `unit`.
    pub data: Vec<f32>,
    pub unit: String,
}

impl SnippetSeries {
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}
