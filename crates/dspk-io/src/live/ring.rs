//! The ring buffer of a running source: its most recent blocks, so views and range reads of the
//! last seconds never touch the disk, and the block being recorded is readable before it is stored.
//!
//! Sized by time with a memory cap ([`LiveRing::capacity_for`]). One writer (the source's thread)
//! pushes; any number of readers copy ranges out. Blocks are shared (`Arc`), so a reader holds the
//! lock only to clone the few it needs.
//!
//! First version: a mutex around a queue. The design (a sequence number per slot, readers that never
//! block the writer) is for when a measurement shows the lock matters.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use super::{LiveBlock, LiveLayout};

/// How full a ring is.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RingStats {
    /// Samples held, of `capacity`.
    pub samples: u64,
    pub capacity: u64,
    /// The first sample still held, and one past the newest.
    pub oldest: u64,
    pub head: u64,
}

impl RingStats {
    /// 0 – 1.
    pub fn use_fraction(&self) -> f32 {
        if self.capacity == 0 { 0.0 } else { (self.samples as f64 / self.capacity as f64) as f32 }
    }
}

pub struct LiveRing {
    layout: LiveLayout,
    capacity: u64,
    blocks: Mutex<VecDeque<Arc<LiveBlock>>>,
}

impl LiveRing {
    /// Samples a ring keeps: `seconds` of signal, but no more than `max_bytes`, and never fewer than
    /// `min_samples` (two chunks of the recorder: the one filling and the one being written).
    pub fn capacity_for(layout: &LiveLayout, seconds: f64, max_bytes: u64, min_samples: u64) -> u64 {
        let by_time = (seconds * layout.sample_rate.rate_hz()).ceil() as u64;
        let by_memory = max_bytes / layout.frame_bytes().max(1) as u64;
        by_time.min(by_memory).max(min_samples).max(1)
    }

    pub fn new(layout: LiveLayout, capacity_samples: u64) -> Self {
        Self { layout, capacity: capacity_samples.max(1), blocks: Mutex::new(VecDeque::new()) }
    }

    pub fn layout(&self) -> &LiveLayout {
        &self.layout
    }

    /// Adds the newest block, dropping the oldest ones the capacity does not need (it keeps at
    /// least `capacity` samples once it has them).
    pub fn push(&self, block: Arc<LiveBlock>) {
        let mut blocks = self.blocks.lock().unwrap_or_else(|e| e.into_inner());
        blocks.push_back(block);
        // Drop the oldest only while what stays still holds the capacity: never fewer samples than asked
        let mut held: u64 = blocks.iter().map(|b| b.samples as u64).sum();
        while let Some(oldest) = blocks.front() {
            let left = held - oldest.samples as u64;
            if left < self.capacity {
                break;
            }
            held = left;
            blocks.pop_front();
        }
    }

    pub fn stats(&self) -> RingStats {
        let blocks = self.blocks.lock().unwrap_or_else(|e| e.into_inner());
        RingStats {
            samples: blocks.iter().map(|b| b.samples as u64).sum(),
            capacity: self.capacity,
            oldest: blocks.front().map_or(0, |b| b.first_sample),
            head: blocks.back().map_or(0, |b| b.end_sample()),
        }
    }

    /// The blocks overlapping `range`, oldest first.
    pub fn blocks_in(&self, range: Range<u64>) -> Vec<Arc<LiveBlock>> {
        let blocks = self.blocks.lock().unwrap_or_else(|e| e.into_inner());
        blocks.iter().filter(|b| b.first_sample < range.end && b.end_sample() > range.start).cloned().collect()
    }

    /// Stored values of `channels` over `range`, channel-major, if the ring holds all of it with no
    /// gap; `None` when part of it is older than the ring (read it from the disk) or not acquired yet.
    pub fn read_stored(&self, channels: &[usize], range: Range<u64>) -> Option<Vec<u8>> {
        let bytes = self.layout.format.bytes();
        let n = range.end.checked_sub(range.start)? as usize;
        let blocks = self.blocks_in(range.clone());
        // Covered without holes?
        let mut at = range.start;
        for b in &blocks {
            if b.first_sample > at {
                return None;
            }
            at = at.max(b.end_sample());
        }
        if at < range.end {
            return None;
        }
        let mut out = vec![0u8; channels.len() * n * bytes];
        for b in &blocks {
            let from = range.start.max(b.first_sample);
            let to = range.end.min(b.end_sample());
            let len = (to - from) as usize * bytes;
            let src_col = (from - b.first_sample) as usize * bytes;
            let dst_col = (from - range.start) as usize * bytes;
            for (row, &c) in channels.iter().enumerate() {
                let src = c * b.samples * bytes + src_col;
                let dst = row * n * bytes + dst_col;
                out[dst..dst + len].copy_from_slice(&b.data[src..src + len]);
            }
        }
        Some(out)
    }

    /// As [`Self::read_stored`], scaled to each channel's unit.
    pub fn read(&self, channels: &[usize], range: Range<u64>) -> Option<Vec<f32>> {
        let n = (range.end - range.start) as usize;
        let stored = self.read_stored(channels, range)?;
        let row_bytes = n * self.layout.format.bytes();
        let mut out = vec![0.0f32; channels.len() * n];
        for (row, &c) in channels.iter().enumerate() {
            let ch = &self.layout.channels[c];
            self.layout.format.decode(&stored[row * row_bytes..(row + 1) * row_bytes], &mut out[row * n..(row + 1) * n], ch.gain, ch.offset);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::encode_stored;
    use dspk_core::{ChannelInfo, SampleFormat, SampleRate, SignalUnit};

    fn layout() -> LiveLayout {
        let channels = (0..2).map(|i| ChannelInfo { name: format!("ch{i}"), gain: 1.0, offset: 0.0, unit: SignalUnit::Microvolt }).collect();
        LiveLayout { name: "r".into(), channels, sample_rate: SampleRate::new(100.0).unwrap(), format: SampleFormat::I16 }
    }

    fn block(l: &LiveLayout, first: u64, samples: usize) -> Arc<LiveBlock> {
        let values: Vec<f32> = (0..2).flat_map(|c| (0..samples).map(move |s| (c * 1000 + first as usize + s) as f32)).collect();
        let data = encode_stored(&values, &l.channels, samples, l.format).unwrap();
        Arc::new(LiveBlock { first_sample: first, samples, source_time_ns: 0, received_unix_ns: 0, lost_before: 0, data })
    }

    #[test]
    fn keeps_the_newest_and_reads_across_blocks() {
        let l = layout();
        let ring = LiveRing::new(l.clone(), 25);
        for k in 0..5 {
            ring.push(block(&l, k * 10, 10));
        }
        let s = ring.stats();
        assert_eq!((s.oldest, s.head), (20, 50));
        assert!(ring.read(&[0], 10..30).is_none(), "older than the ring");
        assert!(ring.read(&[0], 45..55).is_none(), "not acquired yet");
        assert_eq!(ring.read(&[1, 0], 28..32).unwrap(), vec![1028.0, 1029.0, 1030.0, 1031.0, 28.0, 29.0, 30.0, 31.0]);
    }

    #[test]
    fn capacity_by_time_and_memory() {
        let l = layout();
        assert_eq!(LiveRing::capacity_for(&l, 30.0, u64::MAX, 0), 3000);
        // 2 channels × 2 bytes: 400 bytes hold 100 samples
        assert_eq!(LiveRing::capacity_for(&l, 30.0, 400, 0), 100);
        assert_eq!(LiveRing::capacity_for(&l, 30.0, 400, 500), 500);
    }
}
