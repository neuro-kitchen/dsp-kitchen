//! Where every bucket of a pyramid lives, and how the recording is split into pages.
//!
//! Level 0 holds `[min, max]` of every `base` consecutive samples per channel; each next level
//! halves the bucket count, down to one bucket for the whole recording. Level `k` is stored
//! `[channel][bucket]` as `[f32; 2]`, levels one after the other from `header_bytes`.
//!
//! Pages are the unit of filling: `base · 2^page_level` samples (about [`PAGE_SEC`]), so every
//! level up to `page_level` has whole buckets inside each page, and coarser levels merge whole
//! pages.

/// Seconds of recording per page (rounded up to a power-of-two number of level-0 buckets).
pub const PAGE_SEC: f64 = 1.0;

/// Bytes of one `[min, max]` bucket.
const BUCKET_BYTES: usize = std::mem::size_of::<[f32; 2]>();

#[derive(Debug, Clone, Copy)]
pub(crate) struct Level {
    /// Samples per bucket.
    pub bucket: u64,
    pub buckets: u64,
    /// Byte offset of the level's `[channel][bucket]` data.
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct Layout {
    pub channels: usize,
    pub samples: u64,
    pub base: u64,
    pub levels: Vec<Level>,
    /// Finest level whose bucket is a whole page.
    pub page_level: usize,
    pub page_samples: u64,
    pub pages: u64,
    /// Bytes of the header and every level.
    pub bytes: usize,
}

impl Layout {
    /// The layout of a `channels × samples` recording at `sample_rate_hz` with level-0 buckets of
    /// `base` samples, its data starting at `header_bytes`.
    pub fn new(channels: usize, samples: u64, sample_rate_hz: f64, base: u64, header_bytes: usize) -> Self {
        let mut levels = Vec::new();
        let mut offset = header_bytes;
        let (mut bucket, mut buckets) = (base, samples.div_ceil(base).max(1));
        loop {
            levels.push(Level { bucket, buckets, offset });
            offset += channels * buckets as usize * BUCKET_BYTES;
            if buckets == 1 {
                break;
            }
            bucket *= 2;
            buckets = buckets.div_ceil(2);
        }
        let page_hint = (sample_rate_hz * PAGE_SEC).ceil().max(1.0) as u64;
        let page_level = levels.iter().position(|l| l.bucket >= page_hint).unwrap_or(levels.len() - 1);
        let page_samples = levels[page_level].bucket;
        Self { channels, samples, base, page_level, page_samples, pages: samples.div_ceil(page_samples), levels, bytes: offset }
    }

    /// Byte offset of bucket `bucket` of `channel` at level `level`.
    pub fn slot(&self, level: usize, channel: usize, bucket: u64) -> usize {
        let l = &self.levels[level];
        l.offset + (channel * l.buckets as usize + bucket as usize) * BUCKET_BYTES
    }

    /// Samples `start..end` of page `page`.
    pub fn page_range(&self, page: u64) -> std::ops::Range<u64> {
        let start = page * self.page_samples;
        start..(start + self.page_samples).min(self.samples)
    }

    /// Buckets of `level` holding samples `range` (`range` aligned to that level's buckets at its
    /// start).
    pub fn buckets_of(&self, level: usize, range: &std::ops::Range<u64>) -> std::ops::Range<u64> {
        let bucket = self.levels[level].bucket;
        range.start / bucket..range.end.div_ceil(bucket)
    }

    /// Finest level with at most one bucket per column of a `width`-column view of `n` samples
    /// (`None` when columns are finer than `base`: read those windows raw).
    pub fn view_level(&self, n: u64, width: usize) -> Option<usize> {
        if width == 0 || n < self.base * width as u64 {
            return None;
        }
        self.levels.iter().rposition(|l| l.bucket * width as u64 <= n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_hold_whole_buckets_of_every_level_up_to_the_page_level() {
        let l = Layout::new(3, 100_003, 1000.0, 16, 0);
        assert_eq!(l.page_samples, 1024, "≥ 1 s rounded up to 16·2^k");
        assert_eq!(l.levels[l.page_level].bucket, l.page_samples);
        assert_eq!(l.pages, 98);
        assert_eq!(l.levels.last().unwrap().buckets, 1);
        assert_eq!(l.buckets_of(0, &l.page_range(97)), 6208..6251, "the last page is short");
        assert_eq!(l.view_level(100_003, 97), Some(6), "1024-sample buckets: 1 ≤ per column < 2");
        assert_eq!(l.view_level(1_000, 100), None, "finer than base");
    }

    #[test]
    fn a_recording_shorter_than_a_page_is_one_page() {
        let l = Layout::new(1, 100, 30_000.0, 64, 0);
        assert_eq!((l.pages, l.page_level, l.levels.len()), (1, 1, 2));
    }
}
