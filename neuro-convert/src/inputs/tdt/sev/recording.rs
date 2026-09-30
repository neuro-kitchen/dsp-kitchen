//! A stream stored as SEV files (one per channel, optionally split into hour files) as a
//! [`Recording`]. Files are memory-mapped; a sample index maps to (hour file, offset).
//! Rawpacked stores become two recordings: single-unit (high 16 bits) and LFP (low 16 bits).

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use super::files::SevFile;
use super::header::HEADER_BYTES;
use crate::common::codec::decode_into;
use crate::common::mapped::MappedFile;
use crate::error::{Error, Result};
use crate::inputs::tdt::codes;
use crate::model::{check_read, ChannelInfo, MemoryOrder, Recording, RecordingInfo, SampleType, SignalKind};

/// Which part of each stored item a recording exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Whole,
    /// Rawpacked single-unit samples: signed high 16 bits.
    High16,
    /// Rawpacked LFP samples: low 16 bits as signed.
    Low16,
}

/// One channel's data: hour files back to back.
struct ChannelFiles {
    maps: Vec<Arc<MappedFile>>,
    /// First sample of each file, plus the total at the end.
    starts: Vec<u64>,
}

pub struct SevStream {
    info: RecordingInfo,
    channels: Vec<ChannelFiles>,
    /// Stored item type (rawpacked words are read as i32).
    item: SampleType,
    part: Part,
}

impl SevStream {
    /// Builds the recording(s) of one store; rawpacked yields `[single-unit, LFP]`.
    pub fn build(
        store: &str,
        files: &BTreeMap<u16, Vec<SevFile>>,
        start_time: f64,
        expected_rate: Option<f64>,
        rawpacked: bool,
        description: String,
        warnings: &mut Vec<String>,
    ) -> Result<Vec<SevStream>> {
        let first = files.values().flat_map(|v| v.first()).next().ok_or_else(|| Error::format("tdt-sev", format!("{store}: no files")))?;
        let header = &first.header;
        if header.version == 0 {
            warnings.push(format!("{store}: SEV files have no header (v0); assuming float32 at {} Hz", header.sample_rate));
        }
        let item = if rawpacked {
            SampleType::I32
        } else {
            codes::sample_type(header.format as u32).ok_or_else(|| Error::Unsupported(format!("{store}: SEV data format {}", header.format)))?
        };
        let mut rate = header.sample_rate;
        if let Some(expected) = expected_rate.filter(|e| (e - rate).abs() > 1.0) {
            warnings.push(format!("{store}: SEV header rate {rate:.4} Hz differs from the block notes ({expected:.4} Hz); using the block notes"));
            rate = expected;
        }

        let mut channels = Vec::new();
        for (chan, hours) in files {
            if !rawpacked && hours.iter().any(|f| f.header.format != header.format) {
                return Err(Error::format("tdt-sev", format!("{store} ch{chan}: data format changes between files")));
            }
            let mut maps = Vec::new();
            let mut starts = vec![0u64];
            for f in hours {
                maps.push(Arc::new(MappedFile::open(&f.path)?));
                let last = *starts.last().unwrap();
                starts.push(last + f.data_bytes / item.bytes() as u64);
            }
            channels.push(ChannelFiles { maps, starts });
        }
        let per_channel = channels.iter().map(|c| *c.starts.last().unwrap()).min().unwrap_or(0);
        if channels.iter().any(|c| *c.starts.last().unwrap() != per_channel) {
            warnings.push(format!("{store}: SEV channels have different lengths; truncated to {per_channel} samples"));
        }

        let mut metadata = BTreeMap::new();
        metadata.insert("tdt_store".into(), store.to_string());
        metadata.insert("tdt_storage".into(), "sev".into());
        metadata.insert("sev_version".into(), header.version.to_string());
        let hours = files.values().map(Vec::len).max().unwrap_or(1);
        if hours > 1 {
            metadata.insert("sev_hour_files".into(), hours.to_string());
        }
        let make = |name: String, part: Part, stored: SampleType, desc: String| RecordingInfo {
            channels: files.keys().map(|c| ChannelInfo::unity(format!("{name} {c}"))).collect(),
            name,
            description: desc,
            samples: per_channel,
            sample_rate: rate,
            start_time,
            unit: if matches!(stored, SampleType::F32 | SampleType::F64) { "V".into() } else { "a.u.".into() },
            kind: SignalKind::Other,
            stored_as: stored,
            order: MemoryOrder::ChannelMajor,
            metadata: {
                let mut m = metadata.clone();
                if part != Part::Whole {
                    m.insert("rawpacked_part".into(), if part == Part::High16 { "single-unit".into() } else { "lfp".into() });
                }
                m
            },
        };

        if !rawpacked {
            return Ok(vec![SevStream { info: make(store.to_string(), Part::Whole, item, description), channels, item, part: Part::Whole }]);
        }
        // Rawpacked: share the mapped files between the two views
        let dup = |c: &ChannelFiles| ChannelFiles { maps: c.maps.clone(), starts: c.starts.clone() };
        let lfp_channels: Vec<ChannelFiles> = channels.iter().map(dup).collect();
        Ok(vec![
            SevStream { info: make(format!("{store}_SU"), Part::High16, SampleType::I16, format!("{description} (single-unit band)")), channels, item, part: Part::High16 },
            SevStream { info: make(format!("{store}_LFP"), Part::Low16, SampleType::I16, format!("{description} (LFP band)")), channels: lfp_channels, item, part: Part::Low16 },
        ])
    }

    /// Calls `f(bytes)` for each contiguous stored run of `samples` of channel `ch`.
    fn runs(&self, ch: usize, samples: Range<u64>, mut f: impl FnMut(u64, &[u8])) {
        let c = &self.channels[ch];
        let bps = self.item.bytes() as u64;
        let mut s = samples.start;
        while s < samples.end {
            let file = c.starts.partition_point(|&st| st <= s) - 1;
            let within = s - c.starts[file];
            let len = (c.starts[file + 1] - s).min(samples.end - s);
            let at = HEADER_BYTES + (within * bps) as usize;
            f(s - samples.start, &c.maps[file].bytes()[at..at + (len * bps) as usize]);
            s += len;
        }
    }
}

fn unpack(part: Part, word: &[u8]) -> i16 {
    let w = i32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    match part {
        Part::High16 => (w >> 16) as i16,
        _ => (w & 0xFFFF) as u16 as i16,
    }
}

impl Recording for SevStream {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> Result<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        if n == 0 {
            return Ok(());
        }
        for (dst, &ch) in out.chunks_exact_mut(n).zip(channels) {
            let c = &self.info.channels[ch];
            self.runs(ch, samples.clone(), |d0, bytes| {
                let d0 = d0 as usize;
                match self.part {
                    Part::Whole => {
                        let len = bytes.len() / self.item.bytes();
                        decode_into(self.item, bytes, &mut dst[d0..d0 + len], c.gain, c.offset);
                    }
                    part => {
                        for (i, w) in bytes.chunks_exact(4).enumerate() {
                            dst[d0 + i] = (unpack(part, w) as f64 * c.gain + c.offset) as f32;
                        }
                    }
                }
            });
        }
        Ok(())
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> Result<bool> {
        let bps = self.info.stored_as.bytes();
        let n = check_read(&self.info, channels, &samples, out.len() / bps.max(1))?;
        if out.len() != channels.len() * n * bps {
            return Err(Error::BufferSize { expected: channels.len() * n * bps, actual: out.len() });
        }
        for (dst, &ch) in out.chunks_exact_mut((n * bps).max(1)).zip(channels) {
            self.runs(ch, samples.clone(), |d0, bytes| {
                let d0 = d0 as usize * bps;
                match self.part {
                    Part::Whole => dst[d0..d0 + bytes.len()].copy_from_slice(bytes),
                    part => {
                        for (i, w) in bytes.chunks_exact(4).enumerate() {
                            dst[d0 + 2 * i..d0 + 2 * i + 2].copy_from_slice(&unpack(part, w).to_le_bytes());
                        }
                    }
                }
            });
        }
        Ok(true)
    }
}
