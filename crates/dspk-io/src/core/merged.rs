//! Several recordings read as one, their channels side by side ([`MergedRecording`]).
//!
//! Each part contributes the channels picked from it, in order; parts must share the sample rate
//! and length. A read touches each part once with the channels it owns. When every part stores the
//! same format, stored reads pass through (integer recordings stay compact for device upload);
//! otherwise the merge stores float32 with unit gain, so stored and scaled reads agree.

use std::ops::Range;

use dspk_core::recording::{check_read, check_read_stored};
use dspk_core::{DspError, DspResult, MemoryOrder, RecordingInfo, RecordingSource, SampleFormat};

/// Channels of several recordings, side by side.
pub struct MergedRecording {
    parts: Vec<Box<dyn RecordingSource>>,
    /// For each merged channel: (part, channel within the part).
    map: Vec<(usize, usize)>,
    info: RecordingInfo,
    /// Every part stores `info.format`, so stored reads pass through.
    stored_passthrough: bool,
}

impl MergedRecording {
    /// Merges `parts`, each with the channels to take from it (`None`: all, in order).
    ///
    /// # Errors
    /// [`DspError::InvalidConfig`] when there are no parts, a part's channel does not exist, or the
    /// parts differ in sample rate or length.
    pub fn new(parts: Vec<(Box<dyn RecordingSource>, Option<Vec<usize>>)>) -> DspResult<Self> {
        let Some((first, _)) = parts.first() else {
            return Err(DspError::InvalidConfig("merge: no recordings".into()));
        };
        let (rate, samples) = (first.info().sample_rate, first.info().samples);
        let format = first.info().format;
        let mut map = Vec::new();
        let mut channels = Vec::new();
        let mut names = Vec::new();
        let mut stored_passthrough = true;
        for (p, (source, picked)) in parts.iter().enumerate() {
            let info = source.info();
            if info.sample_rate != rate || info.samples != samples {
                return Err(DspError::InvalidConfig(format!(
                    "merge: {} has {} samples at {} Hz, {} has {} at {} Hz (parts must match)",
                    info.name,
                    info.samples,
                    info.sample_rate_hz(),
                    first.info().name,
                    samples,
                    rate.rate_hz()
                )));
            }
            stored_passthrough &= info.format == format;
            let picked = picked.clone().unwrap_or_else(|| (0..info.channel_count()).collect());
            for c in picked {
                let ch = info.channels.get(c).ok_or_else(|| {
                    DspError::InvalidConfig(format!("merge: {} has no channel {c} ({} channels)", info.name, info.channel_count()))
                })?;
                map.push((p, c));
                channels.push(ch.clone());
            }
            names.push(info.name.clone());
        }
        let stored_format = if stored_passthrough { format } else { SampleFormat::F32 };
        let mut info = RecordingInfo::new(names.join("+"), channels.len(), samples, rate, stored_format, MemoryOrder::ChannelMajor);
        if !stored_passthrough {
            // Stored reads become the scaled values as float32
            for ch in &mut channels {
                ch.gain = 1.0;
                ch.offset = 0.0;
            }
        }
        info.channels = channels;
        let parts = parts.into_iter().map(|(s, _)| s).collect();
        Ok(Self { parts, map, info, stored_passthrough })
    }

    /// For each part, the merged positions requested from it and their channels within the part.
    fn by_part(&self, channels: &[usize]) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut groups = vec![(Vec::new(), Vec::new()); self.parts.len()];
        for (i, &c) in channels.iter().enumerate() {
            let (p, local) = self.map[c];
            groups[p].0.push(i);
            groups[p].1.push(local);
        }
        groups
    }
}

impl RecordingSource for MergedRecording {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        let mut buf = Vec::new();
        for (p, (positions, local)) in self.by_part(channels).into_iter().enumerate() {
            if positions.is_empty() {
                continue;
            }
            buf.resize(local.len() * n, 0.0f32);
            self.parts[p].read(&local, samples.clone(), &mut buf)?;
            for (k, &i) in positions.iter().enumerate() {
                out[i * n..(i + 1) * n].copy_from_slice(&buf[k * n..(k + 1) * n]);
            }
        }
        Ok(())
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        if !self.stored_passthrough {
            // Float32 with unit gain: the scaled values are the stored ones
            let n = check_read_stored(&self.info, channels, &samples, out.len())?;
            let mut values = vec![0.0f32; channels.len() * n];
            self.read(channels, samples, &mut values)?;
            for (dst, v) in out.chunks_exact_mut(4).zip(values) {
                dst.copy_from_slice(&v.to_le_bytes());
            }
            return Ok(());
        }
        let n = check_read_stored(&self.info, channels, &samples, out.len())?;
        let row = n * self.info.format.bytes();
        let mut buf = Vec::new();
        for (p, (positions, local)) in self.by_part(channels).into_iter().enumerate() {
            if positions.is_empty() {
                continue;
            }
            buf.resize(local.len() * row, 0u8);
            self.parts[p].read_stored(&local, samples.clone(), &mut buf)?;
            for (k, &i) in positions.iter().enumerate() {
                out[i * row..(i + 1) * row].copy_from_slice(&buf[k * row..(k + 1) * row]);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_core::MemoryRecording;

    fn rec(name: &str, channels: usize, samples: usize, offset: f32) -> Box<dyn RecordingSource> {
        let data = (0..channels * samples).map(|i| i as f32 + offset).collect();
        Box::new(MemoryRecording::new(name, data, channels, 1_000.0).unwrap())
    }

    #[test]
    fn channels_from_each_part_side_by_side() {
        let merged = MergedRecording::new(vec![(rec("a", 3, 50, 0.0), Some(vec![2, 0])), (rec("b", 2, 50, 1_000.0), None)]).unwrap();
        assert_eq!(merged.info().channel_count(), 4);
        assert_eq!(merged.info().name, "a+b");
        let mut out = vec![0.0f32; 3 * 10];
        // merged channels: a2, a0, b0, b1 → read b1, a2, b0
        merged.read(&[3, 0, 2], 5..15, &mut out).unwrap();
        let expect = |base: f32, c: usize| (5..15).map(move |t| base + (c * 50 + t) as f32);
        let want: Vec<f32> = expect(1_000.0, 1).chain(expect(0.0, 2)).chain(expect(1_000.0, 0)).collect();
        assert_eq!(out, want);
    }

    #[test]
    fn stored_reads_match_scaled_reads() {
        let merged = MergedRecording::new(vec![(rec("a", 2, 40, 0.0), None), (rec("b", 1, 40, 7.0), None)]).unwrap();
        crate::test_support::assert_stored_matches(&merged, &[2, 0, 1], 3..30);
    }

    #[test]
    fn mismatched_parts_are_refused() {
        let err = MergedRecording::new(vec![(rec("a", 1, 40, 0.0), None), (rec("b", 1, 41, 0.0), None)]).err().unwrap();
        assert!(err.to_string().contains("parts must match"), "{err}");
        let err = MergedRecording::new(vec![(rec("a", 1, 40, 0.0), Some(vec![3]))]).err().unwrap();
        assert!(err.to_string().contains("no channel 3"), "{err}");
    }
}
