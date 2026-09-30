//! Stream stores as [`Recording`]s over TEV packets.
//!
//! Each channel's samples live in fixed-size packets scattered through the TEV (channels of
//! all stores interleave in time order); the TSQ gives every packet's offset. Reading a range
//! decodes only the packets it overlaps, straight from the memory-mapped TEV.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use super::codes;
use super::notes::synapse::StoreDescription;
use super::tsq::{session_time, StoreIndex};
use crate::common::codec::decode_into;
use crate::common::mapped::MappedFile;
use crate::error::{Error, Result};
use crate::model::{check_read, ChannelInfo, MemoryOrder, Recording, RecordingInfo, SampleType, SignalKind};

pub struct TdtStream {
    info: RecordingInfo,
    tev: Arc<MappedFile>,
    /// TEV offsets of each channel's packets (index = channel position in `info`).
    packets: Vec<Vec<u64>>,
    samples_per_packet: u64,
}

impl TdtStream {
    /// Builds a stream from its TSQ index. `block_start` is the block's Unix start time.
    pub fn new(
        store: &StoreIndex,
        tev: Arc<MappedFile>,
        block_start: f64,
        description: Option<&StoreDescription>,
        warnings: &mut Vec<String>,
    ) -> Result<Self> {
        let ty = codes::sample_type(store.format).ok_or_else(|| {
            Error::Unsupported(format!("TDT store {}: data format code {} (e.g. rawpacked)", store.name, store.format))
        })?;
        let samples_per_packet = store.packet_bytes / ty.bytes() as u64;
        if samples_per_packet == 0 {
            return Err(Error::format("tdt", format!("store {} has empty packets", store.name)));
        }

        // Every channel must have the same number of complete packets inside the TEV
        let tev_len = tev.len();
        let mut packets: Vec<Vec<u64>> = Vec::with_capacity(store.packets.len());
        for (chan, offs) in &store.packets {
            let complete = offs.iter().take_while(|&&o| o + store.packet_bytes <= tev_len).count();
            if complete < offs.len() {
                warnings.push(format!("{} ch{chan}: {} packets past the end of the TEV were dropped", store.name, offs.len() - complete));
            }
            packets.push(offs[..complete].to_vec());
        }
        let per_channel = packets.iter().map(Vec::len).min().unwrap_or(0);
        if packets.iter().any(|p| p.len() != per_channel) {
            warnings.push(format!("{}: channels have different packet counts; truncated to {per_channel}", store.name));
            packets.iter_mut().for_each(|p| p.truncate(per_channel));
        }

        let first = store.first_timestamp.values().copied().fold(f64::INFINITY, f64::min);
        let mut metadata = BTreeMap::new();
        metadata.insert("tdt_store".into(), store.name.clone());
        metadata.insert("tdt_storage".into(), "tev".into());
        metadata.insert("tdt_evtype".into(), format!("{:#x}", store.evtype));
        if store.evtype & codes::EVTYPE_UCF != 0 {
            metadata.insert("tdt_unscaled".into(), "true".into());
        }
        let mut desc = String::new();
        if let Some(d) = description {
            desc = format!("{} ({})", d.object, d.object_type);
            if let Some(src) = &d.source {
                desc.push_str(&format!("; {src}"));
            }
            for (k, v) in &d.properties {
                metadata.insert(format!("listing_{}", k.to_lowercase()), v.clone());
            }
        }

        let info = RecordingInfo {
            name: store.name.clone(),
            description: desc,
            channels: store.packets.keys().map(|c| ChannelInfo::unity(format!("{} {c}", store.name))).collect(),
            samples: per_channel as u64 * samples_per_packet,
            sample_rate: store.frequency,
            start_time: if first.is_finite() { session_time(first, block_start).max(0.0) } else { 0.0 },
            // Synapse float streams hold volts ("Unity" scale); integer stores are raw counts
            unit: if ty == SampleType::F32 || ty == SampleType::F64 { "V".into() } else { "a.u.".into() },
            kind: SignalKind::Other,
            stored_as: ty,
            order: MemoryOrder::Packets,
            metadata,
        };
        Ok(Self { info, tev, packets, samples_per_packet })
    }
}

impl Recording for TdtStream {
    fn info(&self) -> &RecordingInfo {
        &self.info
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> Result<()> {
        let n = check_read(&self.info, channels, &samples, out.len())?;
        if n == 0 {
            return Ok(());
        }
        let ty = self.info.stored_as;
        let bps = ty.bytes() as u64;
        let spp = self.samples_per_packet;
        let bytes = self.tev.bytes();

        for (dst, &ch) in out.chunks_exact_mut(n).zip(channels) {
            let c = &self.info.channels[ch];
            let mut s = samples.start;
            while s < samples.end {
                let (k, within) = (s / spp, s % spp);
                let len = (spp - within).min(samples.end - s);
                let at = (self.packets[ch][k as usize] + within * bps) as usize;
                let d0 = (s - samples.start) as usize;
                decode_into(ty, &bytes[at..at + (len * bps) as usize], &mut dst[d0..d0 + len as usize], c.gain, c.offset);
                s += len;
            }
        }
        Ok(())
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> Result<bool> {
        let bps = self.info.stored_as.bytes() as u64;
        let n = check_read(&self.info, channels, &samples, out.len() / bps.max(1) as usize)?;
        if out.len() as u64 != channels.len() as u64 * n as u64 * bps {
            return Err(Error::BufferSize { expected: channels.len() * n * bps as usize, actual: out.len() });
        }
        let spp = self.samples_per_packet;
        let bytes = self.tev.bytes();
        for (dst, &ch) in out.chunks_exact_mut((n as u64 * bps).max(1) as usize).zip(channels) {
            let mut s = samples.start;
            while s < samples.end {
                let (k, within) = (s / spp, s % spp);
                let len = (spp - within).min(samples.end - s);
                let at = (self.packets[ch][k as usize] + within * bps) as usize;
                let d0 = ((s - samples.start) * bps) as usize;
                let nb = (len * bps) as usize;
                dst[d0..d0 + nb].copy_from_slice(&bytes[at..at + nb]);
                s += len;
            }
        }
        Ok(true)
    }
}
