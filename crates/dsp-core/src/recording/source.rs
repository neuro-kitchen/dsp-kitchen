use std::ops::Range;

use super::format::SampleFormat;
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

    /// Reads the stored values (`info().format`, little-endian, before each channel's gain and
    /// offset) of `samples` of every channel in `channels` into `out`, channel-major:
    /// `out` holds `channels.len() · n · format.bytes()` bytes. Lets pipelines move compact integer
    /// samples and scale them on the device.
    ///
    /// The default serves `f32` sources whose channels have unit gain and zero offset through
    /// [`Self::read`]; other sources without a native implementation return
    /// [`DspError::UnsupportedFormat`].
    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        let info = self.info();
        let identity = channels.iter().all(|&c| info.channels.get(c).is_some_and(|ch| ch.gain_uv == 1.0 && ch.offset_uv == 0.0));
        if info.format != SampleFormat::F32 || !identity {
            return Err(DspError::UnsupportedFormat(format!("{}: stored {} reads", info.name, info.format.name())));
        }
        let n = check_read_stored(info, channels, &samples, out.len())?;
        let mut values = vec![0.0f32; channels.len() * n];
        self.read(channels, samples, &mut values)?;
        for (dst, v) in out.chunks_exact_mut(4).zip(values) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }

    fn read_channel(&self, channel: usize, samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        self.read(&[channel], samples, out)
    }

    /// Reads into a newly allocated channel-major [`SignalChunk`].
    fn read_chunk(&self, channels: &[usize], samples: Range<u64>) -> DspResult<SignalChunk> {
        let n = (samples.end.saturating_sub(samples.start)) as usize;
        let mut data = vec![0.0f32; channels.len() * n];
        self.read(channels, samples.clone(), &mut data)?;
        let info = self.info();
        let start = RationalTime::from_samples(samples.start, info.sample_rate)?;
        SignalChunk::new(channels.len(), n, info.sample_rate, start, MemoryOrder::ChannelMajor, data)
    }
}

/// Validates a [`RecordingSource::read_stored`] request (`out_len` in bytes) and returns the
/// samples per channel.
pub fn check_read_stored(info: &RecordingInfo, channels: &[usize], samples: &Range<u64>, out_len: usize) -> DspResult<usize> {
    let bytes = info.format.bytes();
    if out_len % bytes != 0 {
        return Err(DspError::ShapeMismatch { expected: vec![channels.len(), bytes], actual: vec![out_len] });
    }
    check_read(info, channels, samples, out_len / bytes)
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
