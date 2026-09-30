//! Lazy zero-load view over a sample range and/or channel subset of any [`RecordingSource`].

use std::ops::Range;
use std::sync::Arc;

use super::info::RecordingInfo;
use super::source::{check_read, check_read_stored, RecordingSource};
use crate::error::{DspError, DspResult};

/// Zero-copy lazy slice of a [`RecordingSource`] across a sample range `[start..end)`
/// and an optional subset of channels.
///
/// Constructing a `SlicedRecording` performs **zero disk I/O**; reads are translated
/// on the fly to the underlying recording source.
pub struct SlicedRecording {
    parent: Arc<dyn RecordingSource>,
    sample_range: Range<u64>,
    channel_map: Vec<usize>,
    info: RecordingInfo,
}

impl SlicedRecording {
    /// Creates a lazy view over `sample_range` and `channels` (`None` = all channels) of `parent`.
    pub fn new(
        parent: Arc<dyn RecordingSource>,
        sample_range: Range<u64>,
        channels: Option<Vec<usize>>,
    ) -> DspResult<Self> {
        let p_info = parent.info();
        if sample_range.start > sample_range.end || sample_range.end > p_info.samples {
            return Err(DspError::SampleRange {
                start: sample_range.start,
                end: sample_range.end,
                total: p_info.samples,
            });
        }

        let total_ch = p_info.channel_count();
        let channel_map = channels.unwrap_or_else(|| (0..total_ch).collect());
        if let Some(&ch) = channel_map.iter().find(|&&c| c >= total_ch) {
            return Err(DspError::InvalidChannel {
                channel: ch,
                total: total_ch,
            });
        }

        let mut info = p_info.clone();
        info.samples = sample_range.end - sample_range.start;
        info.start_time_sec =
            p_info.start_time_sec + (sample_range.start as f64) / p_info.sample_rate_hz();
        info.channels = channel_map
            .iter()
            .map(|&c| p_info.channels[c].clone())
            .collect();
        info.layout = p_info.layout.as_ref().map(|l| l.select_channels(&channel_map));

        Ok(Self {
            parent,
            sample_range,
            channel_map,
            info,
        })
    }

    /// Global sample offset (`sample_range.start`) relative to the parent recording.
    #[inline]
    pub fn sample_offset(&self) -> u64 {
        self.sample_range.start
    }

    /// Underlying channel mapping relative to the parent recording.
    #[inline]
    pub fn channel_map(&self) -> &[usize] {
        &self.channel_map
    }
}

impl RecordingSource for SlicedRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let _ = check_read(&self.info, channels, &samples, out.len())?;
        let mapped_ch: Vec<usize> = channels.iter().map(|&c| self.channel_map[c]).collect();
        let offset = self.sample_range.start;
        self.parent
            .read(&mapped_ch, (offset + samples.start)..(offset + samples.end), out)
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        let _ = check_read_stored(&self.info, channels, &samples, out.len())?;
        let mapped_ch: Vec<usize> = channels.iter().map(|&c| self.channel_map[c]).collect();
        let offset = self.sample_range.start;
        self.parent.read_stored(&mapped_ch, (offset + samples.start)..(offset + samples.end), out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::MemoryRecording;

    #[test]
    fn test_sliced_recording_translates_samples_and_channels() {
        // 3 channels x 100 samples: value = ch * 1000 + t
        let mut data = vec![0.0f32; 300];
        for c in 0..3 {
            for t in 0..100 {
                data[c * 100 + t] = (c * 1000 + t) as f32;
            }
        }
        let parent: Arc<dyn RecordingSource> =
            Arc::new(MemoryRecording::new("mem", data, 3, 1000.0).unwrap());

        let sliced = SlicedRecording::new(parent, 20..50, Some(vec![2, 0])).unwrap();
        assert_eq!(sliced.info().channel_count(), 2);
        assert_eq!(sliced.info().samples, 30);
        assert!((sliced.info().start_time_sec - 0.020).abs() < 1e-9);

        let mut out = vec![0.0f32; 2 * 5];
        sliced.read(&[0, 1], 10..15, &mut out).unwrap();
        // sliced ch 0 -> parent ch 2 at samples 30..35
        assert_eq!(&out[0..5], &[2030.0, 2031.0, 2032.0, 2033.0, 2034.0]);
        // sliced ch 1 -> parent ch 0 at samples 30..35
        assert_eq!(&out[5..10], &[30.0, 31.0, 32.0, 33.0, 34.0]);
    }
}
