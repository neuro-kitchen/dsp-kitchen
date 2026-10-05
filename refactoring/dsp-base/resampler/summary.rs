//! In-memory min/max summary of one source, filled as views look at it.
//!
//! The recording is split into pages of whole storage chunks (about a second; one second for
//! sources without chunks). The first frame that needs a page reads it once, every channel, and
//! keeps a min/max pyramid of it: [`BASE`]-sample buckets, then pairs of those, up to one bucket
//! per page. Later frames, and every other view of the source, draw zoomed-out windows from the
//! pages without touching the recording. Pages cost about 1/64 of their samples' size.

use std::sync::OnceLock;

use dsp_core::{DspResult, RecordingSource};

use super::minmax::{finish, fold_block, merge, Block, Columns, EMPTY};
use super::read_pipelined;

/// Samples per finest bucket; windows with fewer samples per pixel column read raw samples.
pub const BASE: u64 = 256;

pub struct MinMaxSummary {
    channels: usize,
    samples: u64,
    /// Samples per page (the last page may be shorter).
    page: u64,
    pages: Vec<OnceLock<Page>>,
}

/// Min/max pyramid of one page: `levels[k]` holds `[channel][bucket]` for buckets of
/// `BASE · 2^k` samples, `counts[k]` buckets per channel; the last level has one bucket.
struct Page {
    levels: Vec<Vec<[f32; 2]>>,
    counts: Vec<usize>,
}

impl Page {
    /// Summarizes samples `range` (block-relative) of every channel of `block`.
    fn build(block: &Block, range: std::ops::Range<usize>) -> Self {
        let mut count = range.len().div_ceil(BASE as usize).max(1);
        let mut level = vec![EMPTY; block.channels * count];
        let origin = block.first + range.start as u64;
        fold_block(block, range, Columns::Buckets { origin, size: BASE, count }, &mut level);
        let (mut levels, mut counts) = (vec![level], vec![count]);
        while count > 1 {
            let below = levels.last().unwrap();
            let next_count = count.div_ceil(2);
            let mut next = Vec::with_capacity(block.channels * next_count);
            for row in below.chunks_exact(count) {
                next.extend(row.chunks(2).map(|pair| pair.iter().fold(EMPTY, |acc, &v| merge(acc, v))));
            }
            levels.push(next);
            counts.push(next_count);
            count = next_count;
        }
        Self { levels, counts }
    }
}

impl MinMaxSummary {
    pub fn new(source: &dyn RecordingSource) -> Self {
        let info = source.info();
        let second = info.sample_rate_hz().ceil().max(1.0) as u64;
        let page = match source.chunk_samples().filter(|&c| c > 0) {
            Some(c) => c * (second / c).max(1),
            None => second,
        };
        let count = info.samples.div_ceil(page) as usize;
        Self { channels: info.channel_count(), samples: info.samples, page, pages: (0..count).map(|_| OnceLock::new()).collect() }
    }

    /// Reads the pages covering samples `start..end` that are not summarized yet (every channel,
    /// in the source's native order), consecutive ones together (up to `block_values` values per
    /// read, at least one page), reading ahead while pages are built. Returns `Ok(false)` when
    /// `stop` returned true before a read; pages read until then are kept.
    pub fn fill(&self, source: &dyn RecordingSource, start: u64, end: u64, block_values: usize, stop: impl Fn() -> bool + Sync) -> DspResult<bool> {
        let end = end.min(self.samples);
        if start >= end || self.channels == 0 {
            return Ok(true);
        }
        let (first, last) = ((start / self.page) as usize, end.div_ceil(self.page) as usize);
        let per_read = (block_values / (self.channels * self.page as usize)).max(1);
        let mut ranges = Vec::new();
        let mut p = first;
        while p < last {
            if self.pages[p].get().is_some() {
                p += 1;
                continue;
            }
            let mut q = p + 1;
            while q < last && q - p < per_read && self.pages[q].get().is_none() {
                q += 1;
            }
            ranges.push(p as u64 * self.page..(q as u64 * self.page).min(self.samples));
            p = q;
        }
        read_pipelined(source, ranges, stop, |block| {
            let first_page = (block.first / self.page) as usize;
            let mut offset = 0;
            for i in first_page.. {
                if offset >= block.samples {
                    break;
                }
                let len = (self.page as usize).min(block.samples - offset);
                let _ = self.pages[i].set(Page::build(block, offset..offset + len));
                offset += len;
            }
        })
    }

    /// Whether every page covering samples `start..end` is summarized.
    pub fn covers(&self, start: u64, end: u64) -> bool {
        let end = end.min(self.samples);
        start >= end || self.pages[(start / self.page) as usize..end.div_ceil(self.page) as usize].iter().all(|p| p.get().is_some())
    }

    /// `[min, max]` per pixel column of samples `start..end` for each of `channels`, row by row
    /// into `out` (`channels.len() × width`), from the summarized pages (columns of pages not
    /// read are NaN). Uses the coarsest buckets no longer than a column; each bucket goes to
    /// the column holding its first sample, so column edges snap to bucket edges.
    pub fn envelope(&self, channels: &[usize], start: u64, end: u64, width: usize, out: &mut [[f32; 2]]) {
        let end = end.min(self.samples);
        out.fill(EMPTY);
        if start < end && width > 0 {
            let n = end - start;
            let per_column = n / width as u64;
            let column_of = |s: u64| ((((s - start + 1) * width as u64 - 1) / n) as usize).min(width - 1);
            for p in (start / self.page) as usize..end.div_ceil(self.page) as usize {
                let Some(page) = self.pages[p].get() else { continue };
                let p0 = p as u64 * self.page;
                let mut k = 0;
                while k + 1 < page.levels.len() && BASE << (k + 1) <= per_column {
                    k += 1;
                }
                let (bucket, count, level) = (BASE << k, page.counts[k], &page.levels[k]);
                let j0 = start.saturating_sub(p0) / bucket;
                let j1 = ((end - p0).div_ceil(bucket) as usize).min(count);
                for j in j0 as usize..j1 {
                    let x = column_of((p0 + j as u64 * bucket).max(start));
                    for (r, &ch) in channels.iter().enumerate() {
                        out[r * width + x] = merge(out[r * width + x], level[ch * count + j]);
                    }
                }
            }
        }
        finish(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::MemoryRecording;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn test_summary_matches_bucket_aligned_min_max() {
        let (nch, total, rate) = (2usize, 50_000usize, 1000.0);
        let data: Vec<f32> = (0..nch * total).map(|i| ((i * 7919) % 2003) as f32 - 1000.0).collect();
        let rec = MemoryRecording::new("m", data.clone(), nch, rate).unwrap();
        let summary = MinMaxSummary::new(&rec);
        assert_eq!(summary.page, 1000);

        let (start, end, width) = (1_234u64, 47_000u64, 40usize);
        assert!(summary.fill(&rec, start, end, 1 << 20, || false).unwrap());
        let mut out = vec![[0.0; 2]; width];
        summary.envelope(&[1], start, end, width, &mut out);

        // Every column is the exact min/max of whole buckets around its sample range
        let n = end - start;
        let row = &data[total..];
        let all = row[start as usize..end as usize].iter().fold([f32::INFINITY, f32::NEG_INFINITY], |[a, b], &v| [a.min(v), b.max(v)]);
        let mut seen = [f32::INFINITY, f32::NEG_INFINITY];
        for (x, v) in out.iter().enumerate() {
            assert!(v[0].is_finite(), "column {x} filled");
            let (c0, c1) = (start + x as u64 * n / width as u64, start + (x as u64 + 1) * n / width as u64);
            let slack = 2 * BASE * 8;
            let lo = c0.saturating_sub(slack) as usize;
            let hi = (c1 + slack).min(total as u64) as usize;
            let around = row[lo..hi].iter().fold([f32::INFINITY, f32::NEG_INFINITY], |[a, b], &v| [a.min(v), b.max(v)]);
            assert!(v[0] >= around[0] && v[1] <= around[1], "column {x} stays near its samples");
            seen = [seen[0].min(v[0]), seen[1].max(v[1])];
        }
        assert!(seen[0] <= all[0] && seen[1] >= all[1], "no peak is dropped");
    }

    #[test]
    fn test_fill_stops_on_cancel_and_keeps_pages() {
        let rec = MemoryRecording::new("m", vec![1.0; 10_000], 1, 1000.0).unwrap();
        let summary = MinMaxSummary::new(&rec);
        let cancel = AtomicBool::new(true);
        assert!(!summary.fill(&rec, 0, 10_000, 1 << 20, || cancel.load(Ordering::Relaxed)).unwrap());
        assert!(summary.pages.iter().all(|p| p.get().is_none()));
        cancel.store(false, Ordering::Relaxed);
        assert!(summary.fill(&rec, 0, 5_000, 1000, || cancel.load(Ordering::Relaxed)).unwrap());
        assert_eq!(summary.pages.iter().filter(|p| p.get().is_some()).count(), 5);
        assert!(summary.covers(0, 5_000) && !summary.covers(0, 5_001));
    }
}
