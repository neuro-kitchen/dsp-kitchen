//! A recording read through the app's RAM cache ([`dspk_core::memory::host_cache`]): blocks of
//! decoded samples kept by channels and time, so a view that comes back to a window (a pan back, a
//! zoom in and out, a chain edited) reads RAM instead of decoding the recording's chunks again.
//! Reading, not computing, is what a processed view waits for on a compressed recording.
//!
//! Blocks are aligned to the recording's storage chunks ([`RecordingSource::chunk_samples`]), so
//! each stored chunk is decoded once; the missing blocks of a read are read in one call. Reads
//! larger than [`MAX_CACHED_BYTES`] (a whole-recording pass) go straight to the recording, so a
//! pass does not push out what the views use.

use std::ops::Range;
use std::sync::Arc;

use dspk_core::memory::host_cache;
use dspk_core::{DspResult, RecordingInfo, RecordingSource};

/// Samples per cached block when the recording does not say its chunk size (rounded up to it when
/// it does).
const BLOCK_SAMPLES: u64 = 1 << 16;

/// Reads larger than this are not cached (a view's window of 32 channels over 20 s of int16 is
/// about 31 MiB; tiles of a wide window and a frozen summary's segments are larger).
pub const MAX_CACHED_BYTES: u64 = 32 << 20;

/// `inner` read through the RAM cache, under `prefix` (one per recording).
pub struct CachedSource {
    inner: Arc<dyn RecordingSource>,
    prefix: String,
    block: u64,
}

impl CachedSource {
    pub fn new(inner: Arc<dyn RecordingSource>, prefix: String) -> Self {
        let block = match inner.chunk_samples() {
            Some(c) if c > 0 => BLOCK_SAMPLES.div_ceil(c) * c,
            _ => BLOCK_SAMPLES,
        };
        Self { inner, prefix, block }
    }

    /// `samples` of `channels` (channel-major, `bytes` per value) through the cache: `read` reads a
    /// range of the recording into a buffer, `kind` names what is read (stored or scaled).
    fn cached(&self, kind: &str, channels: &[usize], samples: Range<u64>, bytes: usize, out: &mut [u8], read: impl Fn(Range<u64>, &mut [u8]) -> DspResult<()>) -> DspResult<()> {
        let n = (samples.end - samples.start) as usize;
        let total = self.inner.info().samples;
        if n == 0 {
            return Ok(());
        }
        if (channels.len() * n * bytes) as u64 > MAX_CACHED_BYTES || samples.end > total {
            return read(samples, out);
        }
        let cache = host_cache();
        let key = |b: u64| format!("{}:{kind}:{channels:?}:{b}", self.prefix);
        let (first, last) = (samples.start / self.block, (samples.end - 1) / self.block);
        let mut blocks: Vec<Option<Arc<Vec<u8>>>> = (first..=last).map(|b| cache.get::<Vec<u8>>(&key(b))).collect();
        // The missing blocks, read in one call (from the first missing to the last)
        if let (Some(a), Some(z)) = (blocks.iter().position(Option::is_none), blocks.iter().rposition(Option::is_none)) {
            let (b0, b1) = (first + a as u64, first + z as u64);
            let span = b0 * self.block..((b1 + 1) * self.block).min(total);
            let m = (span.end - span.start) as usize;
            let mut buf = vec![0u8; channels.len() * m * bytes];
            read(span.clone(), &mut buf)?;
            for b in b0..=b1 {
                let r = b * self.block..((b + 1) * self.block).min(total);
                let len = (r.end - r.start) as usize;
                let at = (r.start - span.start) as usize;
                let mut data = Vec::with_capacity(channels.len() * len * bytes);
                for c in 0..channels.len() {
                    let row = c * m + at;
                    data.extend_from_slice(&buf[row * bytes..(row + len) * bytes]);
                }
                let data = Arc::new(data);
                let idx = (b - first) as usize;
                if blocks[idx].is_none() {
                    cache.put(key(b), data.clone(), (channels.len() * len * bytes) as u64);
                    blocks[idx] = Some(data);
                }
            }
        }
        // Each channel's row of `out` from the blocks
        for (i, b) in (first..=last).enumerate() {
            let data = blocks[i].as_ref().expect("every block read");
            let r = b * self.block..((b + 1) * self.block).min(total);
            let len = (r.end - r.start) as usize;
            let (from, to) = (samples.start.max(r.start), samples.end.min(r.end));
            let (a, k) = ((from - r.start) as usize, (to - from) as usize);
            let dst = (from - samples.start) as usize;
            for c in 0..channels.len() {
                let src = c * len + a;
                let d = c * n + dst;
                out[d * bytes..(d + k) * bytes].copy_from_slice(&data[src * bytes..(src + k) * bytes]);
            }
        }
        Ok(())
    }
}

impl RecordingSource for CachedSource {
    fn info(&self) -> &RecordingInfo {
        self.inner.info()
    }

    fn read(&self, channels: &[usize], samples: Range<u64>, out: &mut [f32]) -> DspResult<()> {
        let bytes: &mut [u8] = bytemuck_f32(out);
        self.cached("f32", channels, samples, 4, bytes, |r, buf| {
            let mut values = vec![0f32; buf.len() / 4];
            self.inner.read(channels, r, &mut values)?;
            for (d, v) in buf.chunks_exact_mut(4).zip(values) {
                d.copy_from_slice(&v.to_ne_bytes());
            }
            Ok(())
        })
    }

    fn read_stored(&self, channels: &[usize], samples: Range<u64>, out: &mut [u8]) -> DspResult<()> {
        let bytes = self.inner.info().format.bytes();
        self.cached("stored", channels, samples, bytes, out, |r, buf| self.inner.read_stored(channels, r, buf))
    }

    fn chunk_samples(&self) -> Option<u64> {
        self.inner.chunk_samples()
    }
}

/// `out` as bytes (native order, as the f32 blocks are kept).
fn bytemuck_f32(out: &mut [f32]) -> &mut [u8] {
    // SAFETY: f32 has no invalid bit patterns and u8 has alignment 1; the length is in bytes
    unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(out)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspk_core::MemoryRecording;

    #[test]
    fn reads_through_the_cache_equal_reads_of_the_recording() {
        let (channels, samples) = (3usize, 300_000usize);
        let data: Vec<f32> = (0..channels * samples).map(|i| (i as f32 * 0.37).sin() * 100.0).collect();
        let inner: Arc<dyn RecordingSource> = Arc::new(MemoryRecording::new("cached", data, channels, 30_000.0).unwrap());
        let cached = CachedSource::new(inner.clone(), "test-cached-1".into());
        for (chs, range) in [(vec![0usize, 2], 1000u64..90_000u64), (vec![1], 65_530..65_600), (vec![0, 2], 50_000..140_000), (vec![2, 0, 1], 299_000..300_000), (vec![0, 2], 1000..90_000)] {
            let n = (range.end - range.start) as usize;
            let (mut a, mut b) = (vec![0f32; chs.len() * n], vec![0f32; chs.len() * n]);
            inner.read(&chs, range.clone(), &mut a).unwrap();
            cached.read(&chs, range.clone(), &mut b).unwrap();
            assert_eq!(a, b, "{chs:?} {range:?}");
        }
        assert!(host_cache().used() > 0, "blocks were kept");
        host_cache().forget_prefix("test-cached-1");
    }
}
