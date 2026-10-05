//! Keeps recently decoded chunks of a chunked source in memory.
//!
//! Reading part of a compressed chunk decodes the whole chunk, so views drawing the same region
//! (traces and heatmap), hover readouts and repeated frames over it (panning, resizing) would
//! decode it again and again. [`CachedRecording`] decodes each storage chunk once, all channels,
//! and serves reads from the most recently used chunks within a memory budget. Whole-recording
//! passes ([`RecordingSource::read_native`]) bypass it so they do not evict what views use.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use dsp_core::recording::check_read;
use dsp_core::{DspResult, MemoryOrder, RecordingInfo, RecordingSource};

pub struct CachedRecording {
    inner: Box<dyn RecordingSource>,
    /// Samples per chunk.
    chunk: u64,
    /// Chunks kept.
    capacity: usize,
    /// Decoded chunks, most recently used first: (chunk index, channel-major µV of all channels).
    chunks: Mutex<VecDeque<(u64, Arc<Vec<f32>>)>>,
}

impl CachedRecording {
    /// Wraps `inner` when it is stored in chunks, keeping up to `budget_bytes` of decoded chunks
    /// (at least one); other sources are returned unchanged.
    pub fn wrap(inner: Box<dyn RecordingSource>, budget_bytes: usize) -> Box<dyn RecordingSource> {
        match inner.chunk_samples().filter(|&c| c > 0) {
            Some(chunk) => {
                let bytes = chunk as usize * inner.info().channel_count().max(1) * 4;
                let capacity = (budget_bytes / bytes).max(1);
                Box::new(Self { inner, chunk, capacity, chunks: Mutex::new(VecDeque::new()) })
            }
            None => inner,
        }
    }

    /// Chunk `i` decoded, from memory or read now.
    fn chunk(&self, i: u64) -> DspResult<Arc<Vec<f32>>> {
        {
            let mut chunks = self.chunks.lock().unwrap();
            if let Some(at) = chunks.iter().position(|(c, _)| *c == i) {
                let hit = chunks.remove(at).expect("position is in range");
                chunks.push_front(hit.clone());
                return Ok(hit.1);
            }
        }
        // Decoded outside the lock so other readers are not held up
        let info = self.inner.info();
        let range = i * self.chunk..((i + 1) * self.chunk).min(info.samples);
        let all: Vec<usize> = (0..info.channel_count()).collect();
        let mut data = vec![0.0f32; all.len() * (range.end - range.start) as usize];
        self.inner.read(&all, range, &mut data)?;
        let data = Arc::new(data);
        let mut chunks = self.chunks.lock().unwrap();
        chunks.retain(|(c, _)| *c != i);
        chunks.push_front((i, data.clone()));
        chunks.truncate(self.capacity);
        Ok(data)
    }
}

impl RecordingSource for CachedRecording {
    fn info(&self) -> &RecordingInfo {
        self.inner.info()
    }

    fn chunk_samples(&self) -> Option<u64> {
        Some(self.chunk)
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let n = check_read(self.info(), channels, &samples, out.len())?;
        if n == 0 || channels.is_empty() {
            return Ok(());
        }
        let total = self.info().samples;
        for i in samples.start / self.chunk..samples.end.div_ceil(self.chunk) {
            let data = self.chunk(i)?;
            let (c0, c1) = (i * self.chunk, ((i + 1) * self.chunk).min(total));
            let len = (c1 - c0) as usize;
            let (s0, s1) = (samples.start.max(c0), samples.end.min(c1));
            let (src, dst, count) = ((s0 - c0) as usize, (s0 - samples.start) as usize, (s1 - s0) as usize);
            for (row, &ch) in out.chunks_exact_mut(n).zip(channels) {
                row[dst..dst + count].copy_from_slice(&data[ch * len + src..ch * len + src + count]);
            }
        }
        Ok(())
    }

    fn read_native(&self, samples: Range<u64>, out: &mut [f32]) -> DspResult<MemoryOrder> {
        self.inner.read_native(samples, out)
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        self.inner.read_stored(channels, samples, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::MemoryRecording;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Memory source reporting 100-sample chunks and counting reads.
    struct Chunked(MemoryRecording, Arc<AtomicUsize>);

    impl RecordingSource for Chunked {
        fn info(&self) -> &RecordingInfo {
            self.0.info()
        }
        fn chunk_samples(&self) -> Option<u64> {
            Some(100)
        }
        fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
            self.1.fetch_add(1, Ordering::Relaxed);
            self.0.read(channels, samples, out)
        }
    }

    #[test]
    fn test_reads_match_the_source_and_reuse_chunks() {
        let data: Vec<f32> = (0..3 * 1050).map(|v| v as f32).collect();
        let plain = MemoryRecording::new("m", data.clone(), 3, 1000.0).unwrap();
        let reads = Arc::new(AtomicUsize::new(0));
        // Room for 4 chunks of 3 channels
        let cached = CachedRecording::wrap(Box::new(Chunked(MemoryRecording::new("m", data, 3, 1000.0).unwrap(), reads.clone())), 4 * 100 * 3 * 4);

        for (channels, range) in [(vec![2, 0], 50u64..330), (vec![1], 1000..1050), (vec![0, 1, 2], 120..180)] {
            let (mut a, mut b) = (vec![0.0; channels.len() * (range.end - range.start) as usize], vec![]);
            b.resize(a.len(), 0.0);
            cached.read(&channels, range.clone(), &mut a).unwrap();
            plain.read(&channels, range.clone(), &mut b).unwrap();
            assert_eq!(a, b, "{channels:?} {range:?}");
        }
        // Chunks 0-3, then 10, then 1 again (kept): 5 decodes
        assert_eq!(reads.load(Ordering::Relaxed), 5);
        let mut one = [0.0];
        cached.read(&[0], 0..1, &mut one).unwrap();
        assert_eq!(reads.load(Ordering::Relaxed), 6, "chunk 0 was evicted by the 4-chunk budget");
    }
}
