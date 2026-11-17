//! Live sources: recordings that grow while they are read.
//!
//! A [`LiveSource`] hands out [`LiveBlock`]s in order: a block holds every channel's stored values
//! (the source's [`SampleFormat`], channel-major, before gain and offset) for a run of samples,
//! with the source's own clock at its first sample and the samples lost just before it. A device
//! (`dspk-device`) pulls blocks from each source on a thread of its own, keeps the recent ones in a
//! [`LiveRing`], records them with [`LiveZarrWriter`] and runs its chain on them.
//!
//! Sources here: [`Generator`] (procedural noise, spikes and a sine, paced in real time),
//! [`Paced`] (any recording replayed at its rate) and the minimal sensor format ([`sensor`]).
//!
//! What a run leaves behind besides its data is written by [`SessionLog`] (`session-live.jsonl`).

mod generator;
mod paced;
mod ring;
pub mod sensor;
mod session_log;
#[cfg(all(feature = "zarr", not(target_family = "wasm")))]
mod zarr_append;

use std::time::Duration;

use dspk_core::{ChannelInfo, DspError, DspResult, SampleFormat, SampleRate};
use serde::{Deserialize, Serialize};

pub use generator::{Generator, GeneratorParams};
pub use paced::Paced;
pub use ring::{LiveRing, RingStats};
pub use session_log::{SessionEntry, SessionLog, SESSION_LOG_FILE};
#[cfg(all(feature = "zarr", not(target_family = "wasm")))]
pub use zarr_append::{LiveZarrWriter, RAW_ZARR};

/// What a live source delivers: its channels, rate and stored type. Fixed for the source's life.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveLayout {
    pub name: String,
    pub channels: Vec<ChannelInfo>,
    pub sample_rate: SampleRate,
    /// The stored values' type (int16, int32, float32, …).
    pub format: SampleFormat,
}

impl LiveLayout {
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Bytes of one sample of every channel.
    pub fn frame_bytes(&self) -> usize {
        self.channels.len() * self.format.bytes()
    }

    /// Bytes per second of signal.
    pub fn bytes_per_second(&self) -> f64 {
        self.frame_bytes() as f64 * self.sample_rate.rate_hz()
    }
}

/// A run of samples of every channel, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveBlock {
    /// Index of the block's first sample since the source started (the key of everything).
    pub first_sample: u64,
    /// Samples per channel.
    pub samples: usize,
    /// The source's clock at `first_sample` (ns, the source's own epoch). Sources without a clock
    /// count it from their sample index.
    pub source_time_ns: u64,
    /// When the block arrived, on the device's clock (Unix ns).
    pub received_unix_ns: u64,
    /// Samples the source lost just before `first_sample` (0 = none); `first_sample` already
    /// counts them.
    pub lost_before: u64,
    /// Stored values, channel-major: `channels × samples` of the layout's format, little-endian.
    pub data: Vec<u8>,
}

impl LiveBlock {
    /// One past the block's last sample.
    pub fn end_sample(&self) -> u64 {
        self.first_sample + self.samples as u64
    }

    /// The block's values scaled by each channel's gain and offset, channel-major.
    pub fn to_f32(&self, layout: &LiveLayout) -> Vec<f32> {
        let mut out = vec![0.0f32; layout.channel_count() * self.samples];
        let row_bytes = self.samples * layout.format.bytes();
        for (c, ch) in layout.channels.iter().enumerate() {
            let src = &self.data[c * row_bytes..(c + 1) * row_bytes];
            layout.format.decode(src, &mut out[c * self.samples..(c + 1) * self.samples], ch.gain, ch.offset);
        }
        out
    }

    /// Checks the data's size against `layout`.
    pub fn check(&self, layout: &LiveLayout) -> DspResult<()> {
        let expected = layout.frame_bytes() * self.samples;
        if self.data.len() != expected {
            return Err(DspError::ShapeMismatch { expected: vec![expected], actual: vec![self.data.len()] });
        }
        Ok(())
    }
}

/// What [`LiveSource::next`] found.
#[derive(Debug)]
pub enum LiveEvent {
    Block(LiveBlock),
    /// Nothing arrived within the wait (a silent or unplugged sensor): the caller decides how long
    /// is too long.
    Idle,
    /// The source has nothing more (a replay at its end, a closed connection that will not come back).
    Ended,
}

/// Counts a source keeps about itself (a device reports them as the source's status).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SourceCounters {
    /// Samples lost (gaps the source detected: sequence numbers, missing packets).
    pub lost_samples: u64,
    /// Packets refused for a wrong checksum.
    pub checksum_errors: u64,
    /// Packets refused for any other reason (a wrong header, a layout that changed).
    pub refused_packets: u64,
}

/// A recording that grows while it is read. Pulled by one thread.
pub trait LiveSource: Send {
    fn layout(&self) -> &LiveLayout;

    /// Waits up to `wait` for the next block.
    fn next(&mut self, wait: Duration) -> DspResult<LiveEvent>;

    fn counters(&self) -> SourceCounters {
        SourceCounters::default()
    }
}

/// The device's clock (Unix ns).
pub fn unix_now_ns() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

/// The time of sample `index` at `rate`, in ns from sample 0.
pub fn sample_time_ns(index: u64, rate: SampleRate) -> u64 {
    (index as f64 * 1e9 / rate.rate_hz()).round() as u64
}

/// Samples per block for blocks of `block_ms` at `rate` (at least one).
pub fn block_samples(block_ms: f64, rate: SampleRate) -> usize {
    ((block_ms / 1000.0 * rate.rate_hz()).round() as usize).max(1)
}

/// Stores `values` (scaled, channel-major) as `format` with every channel's gain and offset
/// undone: the inverse of [`LiveBlock::to_f32`]. Integers are rounded and clamped to their range.
pub fn encode_stored(values: &[f32], channels: &[ChannelInfo], samples: usize, format: SampleFormat) -> DspResult<Vec<u8>> {
    let mut out = Vec::with_capacity(values.len() * format.bytes());
    for (c, ch) in channels.iter().enumerate() {
        let gain = if ch.gain == 0.0 { 1.0 } else { ch.gain };
        for &v in &values[c * samples..(c + 1) * samples] {
            let s = (v - ch.offset) / gain;
            match format {
                SampleFormat::I16 => out.extend_from_slice(&(s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16).to_le_bytes()),
                SampleFormat::I32 => out.extend_from_slice(&(s.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32).to_le_bytes()),
                SampleFormat::F32 => out.extend_from_slice(&s.to_le_bytes()),
                other => return Err(DspError::UnsupportedFormat(format!("live sources store int16, int32 or float32, not {}", other.name()))),
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_core::SignalUnit;

    fn layout(format: SampleFormat, gain: f32) -> LiveLayout {
        let channels = (0..2).map(|i| ChannelInfo { name: format!("ch{i}"), gain, offset: 0.0, unit: SignalUnit::Microvolt }).collect();
        LiveLayout { name: "t".into(), channels, sample_rate: SampleRate::new(1000.0).unwrap(), format }
    }

    #[test]
    fn encode_then_decode_round_trips() {
        for (format, gain) in [(SampleFormat::I16, 0.5f32), (SampleFormat::I32, 1.0), (SampleFormat::F32, 1.0)] {
            let l = layout(format, gain);
            let values = vec![1.0, -2.0, 3.5, 100.0, 0.0, -0.5];
            let data = encode_stored(&values, &l.channels, 3, format).unwrap();
            let block = LiveBlock { first_sample: 0, samples: 3, source_time_ns: 0, received_unix_ns: 0, lost_before: 0, data };
            block.check(&l).unwrap();
            assert_eq!(block.to_f32(&l), values, "{format:?}");
        }
    }

    #[test]
    fn block_lengths_follow_time() {
        let rate = SampleRate::new(30_000.0).unwrap();
        assert_eq!(block_samples(10.0, rate), 300);
        assert_eq!(block_samples(0.001, rate), 1);
        assert_eq!(sample_time_ns(30_000, rate), 1_000_000_000);
    }
}
