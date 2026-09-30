use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

use super::format::SampleFormat;
use crate::buffer::MemoryOrder;
use crate::layout::SensorLayout;
use crate::time::SampleRate;

/// Name and scaling of one stored channel: `value_uv = stored * gain_uv + offset_uv`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelInfo {
    pub name: String,
    pub gain_uv: f32,
    pub offset_uv: f32,
}

/// Format-independent description of a continuous multi-channel recording.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordingInfo {
    pub name: String,
    pub channels: Vec<ChannelInfo>,
    /// Samples per channel.
    pub samples: u64,
    pub sample_rate: SampleRate,
    /// Stored sample type (reads always return µV `f32`).
    pub format: SampleFormat,
    /// Stored memory order (reads always return channel-major).
    pub order: MemoryOrder,
    /// Time of sample 0, in seconds from the acquisition start.
    pub start_time_sec: f64,
    pub layout: Option<SensorLayout>,
    /// Free-form format metadata (e.g. probe type, acquisition software).
    pub metadata: BTreeMap<String, String>,
}

impl RecordingInfo {
    /// Channels named `ch0..chN` with unit gain and no offset.
    pub fn new(
        name: impl Into<String>,
        channel_count: usize,
        samples: u64,
        sample_rate: SampleRate,
        format: SampleFormat,
        order: MemoryOrder,
    ) -> Self {
        let channels = (0..channel_count)
            .map(|i| ChannelInfo { name: format!("ch{i}"), gain_uv: 1.0, offset_uv: 0.0 })
            .collect();
        Self {
            name: name.into(),
            channels,
            samples,
            sample_rate,
            format,
            order,
            start_time_sec: 0.0,
            layout: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Sets the same gain on every channel.
    pub fn with_gain_uv(mut self, gain_uv: f32) -> Self {
        for c in &mut self.channels {
            c.gain_uv = gain_uv;
        }
        self
    }

    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    pub fn sample_rate_hz(&self) -> f64 {
        self.sample_rate.rate_hz()
    }

    pub fn duration_sec(&self) -> f64 {
        self.samples as f64 / self.sample_rate.rate_hz()
    }

    /// Bytes of sample data (excluding any header).
    pub fn data_bytes(&self) -> u64 {
        self.samples * self.channels.len() as u64 * self.format.bytes() as u64
    }
}
