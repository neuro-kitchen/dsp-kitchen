use std::collections::BTreeMap;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use super::sample::{MemoryOrder, SampleType};
use crate::error::{Error, Result};

/// Name and scaling of one stored channel: `value = stored * gain + offset`, in `unit`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelInfo {
    pub name: String,
    pub gain: f64,
    pub offset: f64,
}

impl ChannelInfo {
    pub fn unity(name: impl Into<String>) -> Self {
        Self { name: name.into(), gain: 1.0, offset: 0.0 }
    }
}

/// What a continuous signal measures; decides e.g. ElectricalSeries vs TimeSeries in NWB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalKind {
    /// Voltage from electrodes (neural, EMG, EEG).
    Electrical,
    /// Anything else (pressure, temperature, stimulator monitors, …).
    #[default]
    Other,
}

/// Format-independent description of one continuous multi-channel signal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordingInfo {
    /// Name inside its session (e.g. a TDT store name).
    pub name: String,
    pub description: String,
    pub channels: Vec<ChannelInfo>,
    /// Samples per channel.
    pub samples: u64,
    pub sample_rate: f64,
    /// Seconds from the session start to sample 0.
    pub start_time: f64,
    /// Physical unit after scaling (`V`, `uV`, `a.u.`, …).
    pub unit: String,
    pub kind: SignalKind,
    /// How the source stores samples (reads always return channel-major `f32`).
    pub stored_as: SampleType,
    pub order: MemoryOrder,
    pub metadata: BTreeMap<String, String>,
}

impl RecordingInfo {
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    pub fn duration(&self) -> f64 {
        if self.sample_rate > 0.0 { self.samples as f64 / self.sample_rate } else { 0.0 }
    }

    /// Bytes of stored sample data.
    pub fn stored_bytes(&self) -> u64 {
        self.samples * self.channels.len() as u64 * self.stored_as.bytes() as u64
    }
}

/// Read-only continuous signal, read in bounded chunks so hours-long files never sit in memory.
pub trait Recording: Send + Sync {
    fn info(&self) -> &RecordingInfo;

    /// Reads `samples` of every channel in `channels` into `out`, channel-major and scaled:
    /// `out[i * n..(i + 1) * n]` holds `channels[i]`, where `n = samples.end - samples.start`.
    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> Result<()>;

    /// Reads samples exactly as stored (little-endian `stored_as`, unscaled), channel-major, into
    /// `out` (`channels.len() * n * stored_as.bytes()` bytes). Returns `Ok(false)` when the source
    /// cannot, in which case callers use [`read`](Self::read).
    fn read_stored(&self, _channels: &[usize], _samples: Range<u64>, _out: &mut [u8]) -> Result<bool> {
        Ok(false)
    }
}

/// Validates a [`Recording::read`] request and returns the samples per channel.
pub fn check_read(info: &RecordingInfo, channels: &[usize], samples: &Range<u64>, out_len: usize) -> Result<usize> {
    if samples.start > samples.end || samples.end > info.samples {
        return Err(Error::SampleRange { start: samples.start, end: samples.end, total: info.samples });
    }
    let total = info.channels.len();
    if let Some(&channel) = channels.iter().find(|&&c| c >= total) {
        return Err(Error::Channel { channel, total });
    }
    let n = (samples.end - samples.start) as usize;
    if out_len != channels.len() * n {
        return Err(Error::BufferSize { expected: channels.len() * n, actual: out_len });
    }
    Ok(n)
}
