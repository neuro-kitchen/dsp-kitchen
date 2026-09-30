use std::ops::Range;

use super::info::RecordingInfo;
use crate::buffer::{MemoryOrder, SignalChunk};
use crate::error::{DspError, DspResult};
use crate::time::RationalTime;

/// Read-only, thread-shareable continuous recording, read in bounded chunks.
///
/// Implementors never need to hold the whole recording in memory: callers ask for the
/// channels and sample range they draw or process.
pub trait RecordingSource: Send + Sync {
    fn info(&self) -> &RecordingInfo;

    /// Reads `samples` of every channel in `channels` into `out`, channel-major and in µV:
    /// `out[i * n..(i + 1) * n]` holds `channels[i]`, where `n = samples.end - samples.start`.
    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()>;

    fn read_channel(&self, channel: usize, samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        self.read(&[channel], samples, out)
    }

    /// Reads into a newly allocated channel-major [`SignalChunk`].
    fn read_chunk(&self, channels: &[usize], samples: Range<u64>) -> DspResult<SignalChunk> {
        let n = (samples.end.saturating_sub(samples.start)) as usize;
        let mut data = vec![0.0f32; channels.len() * n];
        self.read(channels, samples.clone(), &mut data)?;
        let info = self.info();
        // RationalTime counts whole-Hz rates; fractional rates (e.g. SpikeGLX 30000.12 Hz) round
        let start = RationalTime::from_samples(samples.start, info.sample_rate_hz().round().max(1.0) as u64)?;
        SignalChunk::new(channels.len(), n, info.sample_rate, start, MemoryOrder::ChannelMajor, data)
    }
}

/// Validates a [`RecordingSource::read`] request and returns the samples per channel.
pub fn check_read(info: &RecordingInfo, channels: &[usize], samples: &Range<u64>, out_len: usize) -> DspResult<usize> {
    if samples.start > samples.end || samples.end > info.samples {
        return Err(DspError::SampleRange { start: samples.start, end: samples.end, total: info.samples });
    }
    let total = info.channels.len();
    if let Some(&channel) = channels.iter().find(|&&c| c >= total) {
        return Err(DspError::InvalidChannel { channel, total });
    }
    let n = (samples.end - samples.start) as usize;
    if out_len != channels.len() * n {
        return Err(DspError::ShapeMismatch { expected: vec![channels.len(), n], actual: vec![out_len] });
    }
    Ok(n)
}
