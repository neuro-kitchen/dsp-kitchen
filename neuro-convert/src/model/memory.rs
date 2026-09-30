use std::collections::BTreeMap;
use std::ops::Range;

use super::recording::{check_read, ChannelInfo, Recording, RecordingInfo, SignalKind};
use super::sample::{MemoryOrder, SampleType};
use crate::error::{Error, Result};

/// Recording held in memory as channel-major, already-scaled samples (tests, derived data).
#[derive(Debug, Clone)]
pub struct MemoryRecording {
    info: RecordingInfo,
    data: Vec<f32>,
}

impl MemoryRecording {
    pub fn new(name: &str, data: Vec<f32>, channels: usize, sample_rate: f64, unit: &str) -> Result<Self> {
        if channels == 0 || data.len() % channels != 0 {
            return Err(Error::BufferSize { expected: channels, actual: data.len() });
        }
        let info = RecordingInfo {
            name: name.into(),
            description: String::new(),
            channels: (0..channels).map(|c| ChannelInfo::unity(format!("ch{c}"))).collect(),
            samples: (data.len() / channels) as u64,
            sample_rate,
            start_time: 0.0,
            unit: unit.into(),
            kind: SignalKind::Other,
            stored_as: SampleType::F32,
            order: MemoryOrder::ChannelMajor,
            metadata: BTreeMap::new(),
        };
        Ok(Self { info, data })
    }
}

impl Recording for MemoryRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> Result<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        let total = self.info.samples as usize;
        for (dst, &ch) in out.chunks_exact_mut(n.max(1)).zip(channels) {
            let base = ch * total + samples.start as usize;
            dst.copy_from_slice(&self.data[base..base + n]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_subset_and_bounds() {
        let data = (0..3).flat_map(|c| (0..4).map(move |s| (c * 10 + s) as f32)).collect();
        let r = MemoryRecording::new("m", data, 3, 1000.0, "V").unwrap();
        let mut out = vec![0.0; 4];
        r.read(&[2, 0], 1..3, &mut out).unwrap();
        assert_eq!(out, vec![21.0, 22.0, 1.0, 2.0]);
        assert!(matches!(r.read(&[3], 0..2, &mut out[..2]), Err(Error::Channel { .. })));
        assert!(matches!(r.read(&[0], 3..6, &mut out[..3]), Err(Error::SampleRange { .. })));
        assert!(matches!(r.read(&[0], 0..3, &mut out), Err(Error::BufferSize { .. })));
    }
}
