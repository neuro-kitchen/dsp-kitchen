use std::ops::Range;

use super::format::SampleFormat;
use super::info::RecordingInfo;
use super::source::{check_read, RecordingSource};
use crate::buffer::MemoryOrder;
use crate::error::{DspError, DspResult};
use crate::time::SampleRate;

/// Recording held in memory as channel-major scaled samples (tests, small derived signals).
#[derive(Debug, Clone)]
pub struct MemoryRecording {
    info: RecordingInfo,
    data: Vec<f32>,
}

impl MemoryRecording {
    pub fn new(name: impl Into<String>, data: Vec<f32>, channels: usize, sample_rate_hz: f64) -> DspResult<Self> {
        if channels == 0 || data.len() % channels != 0 {
            return Err(DspError::ShapeMismatch { expected: vec![channels], actual: vec![data.len()] });
        }
        let samples = (data.len() / channels) as u64;
        let info = RecordingInfo::new(
            name,
            channels,
            samples,
            SampleRate::new(sample_rate_hz)?,
            SampleFormat::F32,
            MemoryOrder::ChannelMajor,
        );
        Ok(Self { info, data })
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

impl RecordingSource for MemoryRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        let total = self.info.samples as usize;
        let start = samples.start as usize;
        for (dst, &ch) in out.chunks_exact_mut(n.max(1)).zip(channels) {
            let base = ch * total + start;
            dst.copy_from_slice(&self.data[base..base + n]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec() -> MemoryRecording {
        // 3 channels x 4 samples, value = ch * 10 + sample
        let data = (0..3).flat_map(|c| (0..4).map(move |s| (c * 10 + s) as f32)).collect();
        MemoryRecording::new("m", data, 3, 1000.0).unwrap()
    }

    #[test]
    fn test_read_subset_channel_major() {
        let r = rec();
        let mut out = vec![0.0; 4];
        r.read(&[2, 0], 1..3, &mut out).unwrap();
        assert_eq!(out, vec![21.0, 22.0, 1.0, 2.0]);

        let chunk = r.read_chunk(&[1], 0..4).unwrap();
        assert_eq!(chunk.as_slice(), &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(r.info().duration_sec(), 0.004);
    }

    #[test]
    fn test_read_rejects_bad_requests() {
        let r = rec();
        let mut out = vec![0.0; 4];
        assert!(matches!(r.read(&[3], 0..4, &mut out), Err(DspError::InvalidChannel { .. })));
        assert!(matches!(r.read(&[0], 2..6, &mut out), Err(DspError::SampleRange { .. })));
        assert!(matches!(r.read(&[0], 0..3, &mut out), Err(DspError::ShapeMismatch { .. })));
        // Empty ranges are valid and write nothing
        r.read(&[0, 1], 2..2, &mut []).unwrap();
    }
}
