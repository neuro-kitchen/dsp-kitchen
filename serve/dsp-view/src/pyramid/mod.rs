//! Min/max pyramids of a whole recording: exact envelopes at any zoom without reading every
//! sample.
//!
//! One [`Pyramid`] serves every case. It is kept either in memory, for one session (base
//! [`MEMORY_BASE`]), or in a file next to the recording, built once and reused (base
//! [`FILE_BASE`]). It is filled page by page (about a second each) in any order: wherever a view
//! looks ([`Pyramid::fill`]), or in the background nearest the view first ([`PyramidBuilder`]).
//! Several threads may fill and read one pyramid at once; each page is built once.
//!
//! An envelope uses the coarsest level with at most one bucket per pixel column, with column
//! edges aligned to its buckets: every sample falls into exactly one column, so no peak is lost
//! (column edges move by less than one column). Columns finer than `base` samples are read raw.
//!
//! Files: `layout.rs` (bucket and page geometry), `storage.rs` (memory or file bytes, the file's
//! identity and path), `builder.rs` (the background filler).

mod builder;
mod layout;
mod storage;

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};

use dsp_core::{DspError, DspResult, RecordingSource};

use crate::envelope::fold::{finish, fold_block, merge, Block, Columns, EMPTY};
use crate::read::read_pipelined;
use layout::Layout;
use storage::{Storage, HEADER_BYTES};

pub use builder::{OnProgress, Progress, PyramidBuilder};
pub use layout::PAGE_SEC;
pub use storage::{pyramid_path, PyramidIdentity};

/// Level-0 bucket (samples) of an in-memory pyramid. The pyramid holds about `4 / base` floats
/// per sample, so 256 costs 1/64 of the recording as f32 (filled regions only).
pub const MEMORY_BASE: u64 = 256;
/// Level-0 bucket (samples) of a file pyramid: finer than [`MEMORY_BASE`] (zooms closer before
/// raw reads take over) at 1/16 of the recording as f32, on disk.
pub const FILE_BASE: u64 = 64;

/// State of a bucket coarser than a page: children still to come (1 or 2), then `MERGING`
/// while the last child's thread merges them, then `DONE`.
const MERGING: u8 = 0;
const DONE: u8 = u8::MAX;

/// A min/max pyramid of one recording, in memory or in a file.
pub struct Pyramid {
    storage: Storage,
    layout: Layout,
    /// One bit per page: taken by the thread building it.
    claimed: Vec<AtomicU64>,
    /// One bit per page: built (published with `Release`).
    ready: Vec<AtomicU64>,
    /// Per level above `layout.page_level`, the state of each bucket.
    upper: Vec<Vec<AtomicU8>>,
    marked_complete: AtomicBool,
}

// SAFETY: page buckets are written only by the thread that claimed the page, before its ready
// bit is set (`Release`); a coarser bucket only by the thread that took it to `MERGING`, before
// `DONE` (`Release`). Readers only read buckets whose bit or state they saw (`Acquire`).
unsafe impl Send for Pyramid {}
unsafe impl Sync for Pyramid {}

fn bits(n: u64) -> Vec<AtomicU64> {
    (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect()
}

impl Pyramid {
    fn with_storage(storage: Storage, layout: Layout, complete: bool) -> Self {
        // One bit per page, all set for a complete pyramid
        let full = |n: u64| -> Vec<AtomicU64> {
            if !complete {
                return bits(n);
            }
            (0..n.div_ceil(64))
                .map(|w| {
                    let valid = (n - 64 * w).min(64);
                    AtomicU64::new(if valid == 64 { u64::MAX } else { (1 << valid) - 1 })
                })
                .collect()
        };
        let upper = (layout.page_level + 1..layout.levels.len())
            .map(|k| {
                let below = layout.levels[k - 1].buckets;
                (0..layout.levels[k].buckets).map(|j| AtomicU8::new(if complete { DONE } else if 2 * j + 1 < below { 2 } else { 1 })).collect()
            })
            .collect();
        Self { claimed: full(layout.pages), ready: full(layout.pages), upper, marked_complete: AtomicBool::new(complete), storage, layout }
    }

    /// An empty pyramid of `source` in memory, with level-0 buckets of `base` samples
    /// ([`MEMORY_BASE`] unless there is a reason otherwise).
    pub fn in_memory(source: &dyn RecordingSource, base: u64) -> DspResult<Self> {
        check_base(base)?;
        let info = source.info();
        let layout = Layout::new(info.channel_count(), info.samples, info.sample_rate_hz(), base, 0);
        Ok(Self::with_storage(Storage::memory(layout.bytes)?, layout, false))
    }

    /// The complete pyramid file at `path` built from `identity` with level-0 bucket `base`;
    /// `None` when it is missing, incomplete or built from something else.
    pub fn open_file(path: &Path, identity: &PyramidIdentity, base: u64) -> DspResult<Option<Self>> {
        check_base(base)?;
        let layout = Layout::new(identity.channels, identity.samples, identity.sample_rate_hz, base, HEADER_BYTES);
        Ok(Storage::open_file(path, identity, base, layout.bytes)?.map(|s| Self::with_storage(s, layout, true)))
    }

    /// A new, empty pyramid file at `path` (replacing any other) to fill.
    pub fn create_file(path: &Path, identity: &PyramidIdentity, base: u64) -> DspResult<Self> {
        check_base(base)?;
        let layout = Layout::new(identity.channels, identity.samples, identity.sample_rate_hz, base, HEADER_BYTES);
        Ok(Self::with_storage(Storage::create_file(path, identity, base, layout.bytes)?, layout, false))
    }

    /// The pyramid of source `source_id` of the recording at `path`, base [`FILE_BASE`]: the
    /// complete file next to it, else a new file there, else (no path, or a folder that cannot
    /// be written) one in memory at [`MEMORY_BASE`]. Returns it and whether it is complete.
    pub fn open_or_create(source: &dyn RecordingSource, recording: Option<(&Path, &str)>) -> DspResult<(Self, bool)> {
        let Some((path, id)) = recording else { return Ok((Self::in_memory(source, MEMORY_BASE)?, false)) };
        let identity = PyramidIdentity::of(path, id, source)?;
        let file = pyramid_path(path, id);
        if let Some(pyramid) = Self::open_file(&file, &identity, FILE_BASE)? {
            return Ok((pyramid, true));
        }
        match Self::create_file(&file, &identity, FILE_BASE) {
            Ok(pyramid) => Ok((pyramid, false)),
            Err(e) => {
                tracing::warn!("cannot write {}: {e}; keeping the min/max pyramid in memory", file.display());
                Ok((Self::in_memory(source, MEMORY_BASE)?, false))
            }
        }
    }

    /// The file behind this pyramid, if any.
    pub fn path(&self) -> Option<&Path> {
        self.storage.path()
    }

    pub fn base(&self) -> u64 {
        self.layout.base
    }

    pub fn samples(&self) -> u64 {
        self.layout.samples
    }

    /// Samples per page (the unit of filling).
    pub fn page_samples(&self) -> u64 {
        self.layout.page_samples
    }

    fn page_ready(&self, page: u64) -> bool {
        self.ready[(page / 64) as usize].load(Ordering::Acquire) & (1 << (page % 64)) != 0
    }

    fn bucket_ready(&self, level: usize, bucket: u64) -> bool {
        if level <= self.layout.page_level {
            self.page_ready(bucket * self.layout.levels[level].bucket / self.layout.page_samples)
        } else {
            self.upper[level - self.layout.page_level - 1][bucket as usize].load(Ordering::Acquire) == DONE
        }
    }

    fn pages_of(&self, start: u64, end: u64) -> std::ops::Range<u64> {
        let end = end.min(self.layout.samples);
        if start >= end { 0..0 } else { start / self.layout.page_samples..end.div_ceil(self.layout.page_samples) }
    }

    /// Whether every page holding samples `start..end` is built.
    pub fn covers(&self, start: u64, end: u64) -> bool {
        self.pages_of(start, end).all(|p| self.page_ready(p))
    }

    /// Whether every bucket of every level is built.
    pub fn is_complete(&self) -> bool {
        let top = self.layout.levels.len() - 1;
        if top > self.layout.page_level { self.bucket_ready(top, 0) } else { self.covers(0, self.layout.samples) }
    }

    fn slot(&self, level: usize, channel: usize, bucket: u64) -> *mut [f32; 2] {
        // SAFETY: `Layout::slot` stays inside the mapping of `layout.bytes`.
        unsafe { self.storage.ptr().add(self.layout.slot(level, channel, bucket)).cast() }
    }

    fn get(&self, level: usize, channel: usize, bucket: u64) -> [f32; 2] {
        // SAFETY: callers read only built buckets (see the `Sync` note).
        unsafe { self.slot(level, channel, bucket).read_unaligned() }
    }

    fn set(&self, level: usize, channel: usize, bucket: u64, value: [f32; 2]) {
        // SAFETY: callers write only buckets they own (see the `Sync` note).
        unsafe { self.slot(level, channel, bucket).write_unaligned(value) }
    }

    /// Merges bucket `bucket` of `level` from its children one level finer.
    fn merge_children(&self, level: usize, bucket: u64, children_end: u64) {
        for ch in 0..self.layout.channels {
            let a = self.get(level - 1, ch, 2 * bucket);
            let v = if 2 * bucket + 1 < children_end { merge(a, self.get(level - 1, ch, 2 * bucket + 1)) } else { a };
            self.set(level, ch, bucket, v);
        }
    }

    /// Builds page `page` from `block` (which holds its samples), unless another thread took it.
    fn build_page(&self, block: &Block, page: u64, scratch: &mut Vec<[f32; 2]>) {
        let (word, bit) = ((page / 64) as usize, 1u64 << (page % 64));
        if self.claimed[word].fetch_or(bit, Ordering::AcqRel) & bit != 0 {
            return;
        }
        let l = &self.layout;
        let range = l.page_range(page);
        let first = l.buckets_of(0, &range);
        let count = (first.end - first.start) as usize;
        scratch.clear();
        scratch.resize(l.channels * count, EMPTY);
        let offset = (range.start - block.first) as usize;
        let columns = Columns::Buckets { origin: range.start, size: l.base, count };
        fold_block(block, offset..offset + (range.end - range.start) as usize, columns, scratch);
        for (ch, row) in scratch.chunks_exact(count).enumerate() {
            // SAFETY: level-0 buckets of this page are contiguous per channel and owned by us.
            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), self.slot(0, ch, first.start), count) };
        }
        for k in 1..=l.page_level {
            let children_end = l.buckets_of(k - 1, &range).end;
            for j in l.buckets_of(k, &range) {
                self.merge_children(k, j, children_end);
            }
        }
        self.ready[word].fetch_or(bit, Ordering::Release);

        // Coarser levels: the last child of a bucket merges it, then reports to its parent
        let mut child = page;
        for k in l.page_level + 1..l.levels.len() {
            let j = child / 2;
            let state = &self.upper[k - l.page_level - 1][j as usize];
            if state.fetch_sub(1, Ordering::AcqRel) != MERGING + 1 {
                return;
            }
            self.merge_children(k, j, l.levels[k - 1].buckets);
            state.store(DONE, Ordering::Release);
            child = j;
        }
    }

    /// Builds the pages holding samples `start..end` that are not built yet, reading every
    /// channel in the source's native order, consecutive pages together (up to `block_values`
    /// values per read, at least one page), reading ahead while pages are built. Returns
    /// `Ok(false)` when `stop` returned true before a read; pages built until then are kept. A
    /// file is marked complete once every page is built.
    pub fn fill(&self, source: &dyn RecordingSource, start: u64, end: u64, block_values: usize, stop: impl Fn() -> bool + Sync) -> DspResult<bool> {
        let info = source.info();
        if info.channel_count() != self.layout.channels || info.samples != self.layout.samples {
            return Err(DspError::InvalidConfig("source does not match the min/max pyramid shape".into()));
        }
        let pages = self.pages_of(start, end);
        if self.layout.channels == 0 || pages.is_empty() {
            return Ok(true);
        }
        let page_values = self.layout.channels * self.layout.page_samples as usize;
        let per_read = (block_values / page_values).max(1) as u64;
        let taken = |p: u64| self.claimed[(p / 64) as usize].load(Ordering::Acquire) & (1 << (p % 64)) != 0;
        let mut ranges = Vec::new();
        let mut p = pages.start;
        while p < pages.end {
            if taken(p) {
                p += 1;
                continue;
            }
            let mut q = p + 1;
            while q < pages.end && q - p < per_read && !taken(q) {
                q += 1;
            }
            ranges.push(self.layout.page_range(p).start..self.layout.page_range(q - 1).end);
            p = q;
        }
        let mut scratch = Vec::new();
        let done = read_pipelined(source, ranges, stop, |block| {
            let end = block.first + block.samples as u64;
            for page in self.pages_of(block.first, end) {
                self.build_page(block, page, &mut scratch);
            }
        })?;
        if self.is_complete() && !self.marked_complete.swap(true, Ordering::AcqRel) {
            self.storage.mark_complete()?;
        }
        Ok(done)
    }

    /// `[min, max]` per pixel column of samples `start..end` for each of `channels`, written
    /// row by row into `out` (`channels.len() × width`). Columns not built yet, or holding only
    /// NaN samples, are NaN. Returns `false` (leaving `out` untouched) when a column spans fewer
    /// than `base` samples: read those windows from the recording.
    pub fn envelope(&self, channels: &[usize], start: u64, end: u64, width: usize, out: &mut [[f32; 2]]) -> DspResult<bool> {
        let l = &self.layout;
        if let Some(&channel) = channels.iter().find(|&&c| c >= l.channels) {
            return Err(DspError::InvalidChannel { channel, total: l.channels });
        }
        if start >= end || end > l.samples || width == 0 {
            return Err(DspError::SampleRange { start, end, total: l.samples });
        }
        if out.len() != channels.len() * width {
            return Err(DspError::ShapeMismatch { expected: vec![channels.len(), width], actual: vec![out.len()] });
        }
        let n = end - start;
        let Some(k) = l.view_level(n, width) else { return Ok(false) };
        let bucket = l.levels[k].bucket;
        let edge = |x: usize| (start + x as u64 * n / width as u64) / bucket;
        for x in 0..width {
            let j0 = edge(x);
            let j1 = if x + 1 == width { end.div_ceil(bucket) } else { edge(x + 1) };
            let built = (j0..j1).all(|j| self.bucket_ready(k, j));
            for (r, &ch) in channels.iter().enumerate() {
                out[r * width + x] = if built { (j0..j1).fold(EMPTY, |acc, j| merge(acc, self.get(k, ch, j))) } else { EMPTY };
            }
        }
        finish(out);
        Ok(true)
    }

    /// Sample range column `x` of a `width`-column envelope of `start..end` covers (the
    /// bucket-aligned edges [`envelope`](Self::envelope) uses), or `None` for raw windows.
    pub fn column_samples(&self, start: u64, end: u64, width: usize, x: usize) -> Option<(u64, u64)> {
        let n = end.checked_sub(start)?;
        let bucket = self.layout.levels[self.layout.view_level(n, width)?].bucket;
        let edge = |x: usize| (start + x as u64 * n / width as u64) / bucket;
        let j1 = if x + 1 == width { end.div_ceil(bucket) } else { edge(x + 1) };
        Some((edge(x) * bucket, (j1 * bucket).min(self.layout.samples)))
    }
}

fn check_base(base: u64) -> DspResult<()> {
    if base == 0 {
        return Err(DspError::InvalidConfig("min/max pyramid base must be at least 1 sample".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::MemoryRecording;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn recording(channels: usize, samples: usize) -> MemoryRecording {
        let data: Vec<f32> = (0..channels * samples)
            .map(|i| {
                let (c, s) = (i / samples, i % samples);
                ((s * 7919 + c * 104_729) % 1001) as f32 - 500.0 + if (s + 37 * c) % 4099 == 0 { -9_000.0 } else { 0.0 }
            })
            .collect();
        MemoryRecording::new("rec", data, channels, 1000.0).unwrap()
    }

    /// Brute-force min/max over each column's bucket-aligned sample range.
    fn expected(rec: &MemoryRecording, p: &Pyramid, channels: &[usize], start: u64, end: u64, width: usize) -> Vec<[f32; 2]> {
        let mut out = vec![[0.0; 2]; channels.len() * width];
        for x in 0..width {
            let (c0, c1) = p.column_samples(start, end, width, x).unwrap();
            let mut buf = vec![0.0f32; channels.len() * (c1 - c0) as usize];
            rec.read(channels, c0..c1, &mut buf).unwrap();
            for (r, row) in buf.chunks_exact((c1 - c0) as usize).enumerate() {
                out[r * width + x] = row.iter().fold([f32::INFINITY, f32::NEG_INFINITY], |[a, b], &v| [a.min(v), b.max(v)]);
            }
        }
        out
    }

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("dk_pyramid_{name}_{}", std::process::id()))
    }

    fn identity(rec: &MemoryRecording) -> PyramidIdentity {
        let info = rec.info();
        PyramidIdentity { source_id: "main".into(), len: 0, mtime_ns: 0, channels: info.channel_count(), samples: info.samples, sample_rate_hz: info.sample_rate_hz() }
    }

    fn check_matches_brute_force(rec: &MemoryRecording, p: &Pyramid) {
        let channels = [4, 0, 2];
        for (start, end, width) in [(0u64, 100_003u64, 97usize), (123, 99_000, 640), (5_000, 7_000, 100), (0, 100_003, 1)] {
            let mut out = vec![[0.0; 2]; channels.len() * width];
            assert!(p.envelope(&channels, start, end, width, &mut out).unwrap());
            assert_eq!(out, expected(rec, p, &channels, start, end, width), "{start}..{end} @ {width}");
            let spikes = (start..end).filter(|s| (*s as usize + 37 * 4).is_multiple_of(4099)).count();
            if spikes > 0 {
                assert!(out[..width].iter().any(|v| v[0] < -8_000.0), "every spike shows");
            }
        }
        let mut out = vec![[0.0; 2]; 3 * 100];
        assert!(!p.envelope(&channels, 0, 1_000, 100, &mut out).unwrap(), "finer than base: raw");
    }

    #[test]
    fn memory_and_file_pyramids_match_brute_force() {
        let rec = recording(5, 100_003);
        let mem = Pyramid::in_memory(&rec, 16).unwrap();
        assert!(mem.fill(&rec, 0, 100_003, 1 << 20, || false).unwrap());
        assert!(mem.is_complete());
        check_matches_brute_force(&rec, &mem);

        let dir = temp_dir("match");
        let file = Pyramid::create_file(&pyramid_path(&dir.join("rec.bin"), "main"), &identity(&rec), 16).unwrap();
        assert!(file.fill(&rec, 0, 100_003, 1 << 20, || false).unwrap());
        check_matches_brute_force(&rec, &file);
        drop(file);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn filling_in_any_order_gives_the_same_pyramid() {
        let rec = recording(5, 100_003);
        let p = Pyramid::in_memory(&rec, 16).unwrap();
        for (s, e) in [(60_000, 70_000), (0, 3_000), (99_000, 100_003), (2_000, 100_003)] {
            assert!(p.fill(&rec, s, e, 4_000, || false).unwrap());
        }
        assert!(p.is_complete());
        check_matches_brute_force(&rec, &p);
    }

    #[test]
    fn concurrent_fills_build_each_page_once_and_agree() {
        let rec = Arc::new(recording(5, 100_003));
        let p = Arc::new(Pyramid::in_memory(rec.as_ref(), 16).unwrap());
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let (rec, p) = (rec.clone(), p.clone());
                std::thread::spawn(move || {
                    let start = t * 20_000;
                    p.fill(rec.as_ref(), start, 100_003, 3_000, || false).unwrap();
                    p.fill(rec.as_ref(), 0, start, 3_000, || false).unwrap();
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert!(p.is_complete());
        check_matches_brute_force(&rec, &p);
    }

    #[test]
    fn unbuilt_columns_are_nan_and_stopping_keeps_built_pages() {
        let rec = recording(2, 50_000);
        let p = Pyramid::in_memory(&rec, 8).unwrap();
        assert!(!p.fill(&rec, 0, 50_000, 1 << 20, || true).unwrap(), "stopped before the first read");
        assert!(!p.covers(0, 1));
        assert!(p.fill(&rec, 0, 20_000, 1 << 20, || false).unwrap());
        assert!(p.covers(0, 20_000) && !p.covers(0, 50_000));
        let mut out = vec![[0.0; 2]; 2 * 50];
        assert!(p.envelope(&[0, 1], 0, 50_000, 50, &mut out).unwrap());
        let built = out[..50].iter().take_while(|v| v[0].is_finite()).count();
        assert!(built > 0 && built < 50 && out[built..50].iter().all(|v| v[0].is_nan()));
    }

    #[test]
    fn files_reopen_only_when_complete_and_unchanged() {
        let rec = recording(2, 50_000);
        let dir = temp_dir("reopen");
        let path = pyramid_path(&dir.join("rec.bin"), "main");
        let id = identity(&rec);

        let p = Pyramid::create_file(&path, &id, 8).unwrap();
        assert!(p.fill(&rec, 0, 20_000, 1 << 20, || false).unwrap());
        drop(p);
        assert!(Pyramid::open_file(&path, &id, 8).unwrap().is_none(), "incomplete files are rebuilt");

        let p = Pyramid::create_file(&path, &id, 8).unwrap();
        assert!(p.fill(&rec, 0, 50_000, 1 << 20, || false).unwrap());
        drop(p);
        let reopened = Pyramid::open_file(&path, &id, 8).unwrap().expect("complete files reopen");
        assert!(reopened.is_complete() && reopened.covers(0, 50_000));
        let mut out = vec![[0.0; 2]; 50];
        assert!(reopened.envelope(&[1], 0, 50_000, 50, &mut out).unwrap());
        assert!(out.iter().all(|v| v[0].is_finite()));
        drop(reopened);
        assert!(Pyramid::open_file(&path, &PyramidIdentity { mtime_ns: 1, ..id.clone() }, 8).unwrap().is_none());
        assert!(Pyramid::open_file(&path, &id, 16).unwrap().is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
