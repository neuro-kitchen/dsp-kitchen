use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

use super::format::SampleFormat;
use super::unit::SignalUnit;
use crate::buffer::MemoryOrder;
use crate::time::{RationalTime, SampleRate};

/// Name and scaling of one stored channel: `value = stored * gain + offset`, in `unit`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelInfo {
    pub name: String,
    pub gain: f32,
    pub offset: f32,
    pub unit: SignalUnit,
}

/// Format-independent description of a continuous multi-channel recording.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordingInfo {
    pub name: String,
    pub channels: Vec<ChannelInfo>,
    /// Samples per channel.
    pub samples: u64,
    pub sample_rate: SampleRate,
    /// Stored sample type (reads always return scaled `f32` in each channel's unit).
    pub format: SampleFormat,
    /// Stored memory order (reads always return channel-major).
    pub order: MemoryOrder,
    /// Time of sample 0 from the acquisition start.
    pub start_time: RationalTime,
    /// Free-form format metadata (e.g. device type, acquisition software).
    pub metadata: BTreeMap<String, String>,
}

impl RecordingInfo {
    /// Channels named `ch0..chN` with unit gain, no offset and dimensionless values.
    pub fn new(
        name: impl Into<String>,
        channel_count: usize,
        samples: u64,
        sample_rate: SampleRate,
        format: SampleFormat,
        order: MemoryOrder,
    ) -> Self {
        let channels = (0..channel_count)
            .map(|i| ChannelInfo { name: format!("ch{i}"), gain: 1.0, offset: 0.0, unit: SignalUnit::Dimensionless })
            .collect();
        Self {
            name: name.into(),
            channels,
            samples,
            sample_rate,
            format,
            order,
            start_time: RationalTime::ZERO,
            metadata: BTreeMap::new(),
        }
    }

    /// Sets the same gain and unit on every channel.
    pub fn with_gain(mut self, gain: f32, unit: SignalUnit) -> Self {
        for c in &mut self.channels {
            c.gain = gain;
            c.unit = unit.clone();
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
