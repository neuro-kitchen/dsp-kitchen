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
use std::sync::Mutex;

use dspk_core::{DspError, DspResult, RecordingSource};

use crate::envelope::fold::{finish, fold_block, merge, Block, Columns, EMPTY};
use crate::read::read_pipelined;
use layout::Layout;
use storage::{data_offset, Storage};

pub use builder::{OnProgress, Progress, PyramidBuilder};
pub use layout::PAGE_SEC;
pub use storage::{cache_dir, drv_pyramid_path, new_uuid, pyramid_path, recording_uuid, PyramidIdentity};

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
    /// Serializes writing a ready word into the mmapped bitmap so concurrent pages in the same
    /// 64-page word never clobber each other's bit.
    bitmap_lock: Mutex<()>,
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
    fn file_layout(channels: usize, samples: u64, sample_rate_hz: f64, base: u64) -> Layout {
        let pages = Layout::new(channels, samples, sample_rate_hz, base, 0).pages;
        Layout::new(channels, samples, sample_rate_hz, base, data_offset(pages))
    }

    fn with_storage(storage: Storage, layout: Layout, ready_words: Option<&[u64]>, complete: bool) -> Self {
        let make_bits = |n: u64| -> Vec<AtomicU64> {
            let words = n.div_ceil(64) as usize;
            if complete {
                return (0..words)
                    .map(|w| {
                        let valid = (n - 64 * w as u64).min(64);
                        AtomicU64::new(if valid == 64 { u64::MAX } else { (1 << valid) - 1 })
                    })
                    .collect();
            }
            if let Some(saved) = ready_words {
                return (0..words)
                    .map(|w| {
                        let valid = (n - 64 * w as u64).min(64);
                        let mask = if valid == 64 { u64::MAX } else { (1 << valid) - 1 };
                        AtomicU64::new(saved.get(w).copied().unwrap_or(0) & mask)
                    })
                    .collect();
            }
            bits(n)
        };
        let upper = (layout.page_level + 1..layout.levels.len())
            .map(|k| {
                let below = layout.levels[k - 1].buckets;
                (0..layout.levels[k].buckets).map(|j| AtomicU8::new(if complete { DONE } else if 2 * j + 1 < below { 2 } else { 1 })).collect()
            })
            .collect();
        let p = Self {
            claimed: make_bits(layout.pages),
            ready: make_bits(layout.pages),
            bitmap_lock: Mutex::new(()),
            upper,
            marked_complete: AtomicBool::new(complete),
            storage,
            layout,
        };
        // Reconstruct coarser levels for a resumed partial file from the persisted ready pages
        if !complete && ready_words.is_some() {
            for k in p.layout.page_level + 1..p.layout.levels.len() {
                let below = p.layout.levels[k - 1].buckets;
                for j in 0..p.layout.levels[k].buckets {
                    let left_ready = p.bucket_ready(k - 1, 2 * j);
                    let has_right = 2 * j + 1 < below;
                    let right_ready = !has_right || p.bucket_ready(k - 1, 2 * j + 1);
                    if left_ready && right_ready {
                        p.merge_children(k, j, below);
                        p.upper[k - p.layout.page_level - 1][j as usize].store(DONE, Ordering::Release);
                    } else {
                        let remaining = (!left_ready as u8) + (has_right && !right_ready) as u8;
                        p.upper[k - p.layout.page_level - 1][j as usize].store(remaining, Ordering::Release);
                    }
                }
            }
            if p.is_complete() && !p.marked_complete.swap(true, Ordering::AcqRel) {
                let _ = p.storage.mark_complete();
            }
        }
        p
    }

    /// An empty pyramid of `source` in memory, with level-0 buckets of `base` samples
    /// ([`MEMORY_BASE`] unless there is a reason otherwise).
    pub fn in_memory(source: &dyn RecordingSource, base: u64) -> DspResult<Self> {
        check_base(base)?;
        let info = source.info();
        let layout = Layout::new(info.channel_count(), info.samples, info.sample_rate_hz(), base, 0);
        Ok(Self::with_storage(Storage::memory(layout.bytes)?, layout, None, false))
    }

    /// An empty pyramid with the given shape in memory (e.g. a derived chain whose channel count
    /// differs from the raw recording's).
    pub fn in_memory_shape(channels: usize, samples: u64, sample_rate_hz: f64, base: u64) -> DspResult<Self> {
        check_base(base)?;
        let layout = Layout::new(channels, samples, sample_rate_hz, base, 0);
        Ok(Self::with_storage(Storage::memory(layout.bytes)?, layout, None, false))
    }

    /// Opens an existing pyramid file at `path` built from `identity` with level-0 bucket `base`
    /// (complete or partial — partial files resume from their persisted ready-page bitmap);
    /// `None` when it is missing or built from a different recording, base, or chain signature.
    pub fn open_file(path: &Path, identity: &PyramidIdentity, base: u64) -> DspResult<Option<Self>> {
        check_base(base)?;
        let layout = Self::file_layout(identity.channels, identity.samples, identity.sample_rate_hz, base);
        let words = layout.pages.div_ceil(64) as usize;
        Ok(Storage::open_file(path, identity, base, layout.bytes, words)?.map(|(s, ready, complete)| {
            Self::with_storage(s, layout, if complete { None } else { Some(&ready) }, complete)
        }))
    }

    /// A new, empty pyramid file at `path` (replacing any other in place) to fill.
    pub fn create_file(path: &Path, identity: &PyramidIdentity, base: u64) -> DspResult<Self> {
        check_base(base)?;
        let layout = Self::file_layout(identity.channels, identity.samples, identity.sample_rate_hz, base);
        let words = layout.pages.div_ceil(64) as usize;
        Ok(Self::with_storage(Storage::create_file(path, identity, base, layout.bytes, words)?, layout, None, false))
    }

    /// Opens or creates/resets the derived-chain pyramid at
    /// `<dir>/cache/<file name>.cache/drv/<drv_id>_drv_<source_id>.minmax` for `identity` (which
    /// carries the chain's signature). Resumes if the file matches `identity`; resets in place if
    /// the chain changed; falls back to memory if the folder cannot be written.
    pub fn open_or_create_drv(path: &Path, source_id: &str, drv_id: &str, identity: &PyramidIdentity, base: u64) -> DspResult<Self> {
        let file = drv_pyramid_path(path, source_id, drv_id);
        if let Some(pyramid) = Self::open_file(&file, identity, base)? {
            return Ok(pyramid);
        }
        match Self::create_file(&file, identity, base) {
            Ok(pyramid) => Ok(pyramid),
            Err(e) => {
                tracing::warn!("cannot write {}: {e}; keeping the derived min/max pyramid in memory", file.display());
                Self::in_memory_shape(identity.channels, identity.samples, identity.sample_rate_hz, base)
            }
        }
    }

    /// The pyramid of source `source_id` of the recording at `path`, base [`FILE_BASE`]: the
    /// existing file next to it (complete or resumable), else a new file there, else (no path, or
    /// a folder that cannot be written) one in memory at [`MEMORY_BASE`]. Returns it and whether
    /// it is complete.
    pub fn open_or_create(source: &dyn RecordingSource, recording: Option<(&Path, &str)>) -> DspResult<(Self, bool)> {
        let Some((path, id)) = recording else { return Ok((Self::in_memory(source, MEMORY_BASE)?, false)) };
        let identity = PyramidIdentity::of(path, id, source)?;
        let file = pyramid_path(path, id);
        if let Some(pyramid) = Self::open_file(&file, &identity, FILE_BASE)? {
            let complete = pyramid.is_complete();
            return Ok((pyramid, complete));
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
        self.publish_page(page, scratch, count);
    }

    /// Writes page `page`'s level-0 buckets (`rows`: `count` per channel, channel after channel),
    /// merges its finer levels and the coarser buckets it completes, and marks it built. The
    /// caller claimed the page.
    fn publish_page(&self, page: u64, rows: &[[f32; 2]], count: usize) {
        let (word, bit) = ((page / 64) as usize, 1u64 << (page % 64));
        let l = &self.layout;
        let range = l.page_range(page);
        let first = l.buckets_of(0, &range);
        for (ch, row) in rows.chunks_exact(count).take(l.channels).enumerate() {
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
        {
            let _g = self.bitmap_lock.lock().unwrap_or_else(|e| e.into_inner());
            let current = self.ready[word].load(Ordering::Acquire);
            self.storage.write_ready_word(word, current);
        }

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
        self.envelope_inner(channels, start, end, width, false, out)
    }

    /// Like [`Self::envelope`], but falls back to level 0 when a column spans fewer than `base`
    /// samples (so zoomed-out views wider than the GPU tile budget never hit a blank dead zone on
    /// high-channel recordings or 4K displays).
    pub fn envelope_clamped(&self, channels: &[usize], start: u64, end: u64, width: usize, out: &mut [[f32; 2]]) -> DspResult<()> {
        self.envelope_inner(channels, start, end, width, true, out).map(|_| ())
    }

    fn envelope_inner(&self, channels: &[usize], start: u64, end: u64, width: usize, clamp: bool, out: &mut [[f32; 2]]) -> DspResult<bool> {
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
        let k = match l.view_level(n, width) {
            Some(k) => k,
            None if clamp => 0,
            None => return Ok(false),
        };
        let bucket = l.levels[k].bucket;
        let max_buckets = l.levels[k].buckets;
        let edge = |x: usize| (start + x as u64 * n / width as u64) / bucket;
        for x in 0..width {
            let mut j0 = edge(x);
            let mut j1 = if x + 1 == width { end.div_ceil(bucket) } else { edge(x + 1) };
            if clamp {
                j0 = j0.min(max_buckets.saturating_sub(1));
                j1 = j1.max(j0 + 1).min(max_buckets);
            }
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

    /// Pages in all (the unit of filling, [`Self::page_samples`] each, the last one shorter).
    pub fn pages(&self) -> u64 {
        self.layout.pages
    }

    /// Whether page `page` is built.
    pub fn page_built(&self, page: u64) -> bool {
        page < self.layout.pages && self.page_ready(page)
    }

    /// Sample range `start..end` covered by page `page`.
    pub fn page_range(&self, page: u64) -> std::ops::Range<u64> {
        self.layout.page_range(page)
    }

    /// Total samples across all ready pages (equals [`Self::samples`] once complete).
    pub fn done_samples(&self) -> u64 {
        (0..self.layout.pages)
            .filter(|&p| self.page_ready(p))
            .map(|p| {
                let r = self.layout.page_range(p);
                r.end - r.start
            })
            .sum()
    }

    /// Level-0 buckets ([`Self::base`] samples each) of page `page`.
    pub fn page_buckets(&self, page: u64) -> usize {
        let range = self.layout.page_range(page);
        let b = self.layout.buckets_of(0, &range);
        (b.end - b.start) as usize
    }

    /// Builds page `page` from its level-0 `[min, max]` buckets, computed elsewhere (a browser
    /// page decodes chunks on several workers and sends only these): `rows` holds
    /// [`Self::page_buckets`] per channel, channel after channel, [`EMPTY`] (`[+∞, −∞]`)
    /// for buckets without samples. Returns `false` when the page was built or taken already, or `rows` has the wrong
    /// length. Fills the same buckets as [`Self::fill`] given the same samples.
    pub fn build_page_from_buckets(&self, page: u64, rows: &[[f32; 2]]) -> DspResult<bool> {
        if page >= self.layout.pages {
            return Ok(false);
        }
        let count = self.page_buckets(page);
        if rows.len() != count * self.layout.channels {
            return Err(DspError::ShapeMismatch { expected: vec![self.layout.channels, count], actual: vec![rows.len()] });
        }
        let (word, bit) = ((page / 64) as usize, 1u64 << (page % 64));
        if self.claimed[word].fetch_or(bit, Ordering::AcqRel) & bit != 0 {
            return Ok(false);
        }
        self.publish_page(page, rows, count);
        if self.is_complete() && !self.marked_complete.swap(true, Ordering::AcqRel) {
            self.storage.mark_complete()?;
        }
        Ok(true)
    }

    /// Whether each of `columns` equal parts of the recording is built (every page in it): a
    /// map of the overview for a progress bar.
    pub fn built_columns(&self, columns: usize) -> Vec<bool> {
        let total = self.layout.samples;
        (0..columns)
            .map(|x| {
                let (a, b) = (total * x as u64 / columns as u64, total * (x as u64 + 1) / columns as u64);
                self.covers(a, b.max(a + 1))
            })
            .collect()
    }

    /// The whole pyramid as bytes, once complete (to keep it: a browser page stores its
    /// in-memory pyramid so the recording opens complete next time); `None` before.
    pub fn complete_bytes(&self) -> Option<Vec<u8>> {
        if !self.is_complete() {
            return None;
        }
        // SAFETY: every bucket is built, and built buckets are never written again.
        Some(unsafe { std::slice::from_raw_parts(self.storage.ptr(), self.layout.bytes) }.to_vec())
    }

    /// Fills this (still empty) pyramid from `bytes` that [`Self::complete_bytes`] gave for the
    /// same recording and base, and marks it complete. Returns `false`, changing nothing, when
    /// the length does not match or a page was taken already.
    pub fn restore(&self, bytes: &[u8]) -> bool {
        if bytes.len() != self.layout.bytes || (0..self.layout.pages).any(|p| self.claimed[(p / 64) as usize].load(Ordering::Acquire) & (1 << (p % 64)) != 0) {
            return false;
        }
        for w in &self.claimed {
            w.store(u64::MAX, Ordering::Release);
        }
        // SAFETY: no page was claimed, so no thread writes or reads a bucket of it; readers only
        // read buckets after their ready bit (set below, `Release`).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.storage.ptr(), bytes.len()) };
        let n = self.layout.pages;
        for (w, word) in self.ready.iter().enumerate() {
            let valid = (n - 64 * w as u64).min(64);
            word.store(if valid == 64 { u64::MAX } else { (1 << valid) - 1 }, Ordering::Release);
        }
        for level in &self.upper {
            for state in level {
                state.store(DONE, Ordering::Release);
            }
        }
        self.marked_complete.store(true, Ordering::Release);
        true
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
    use dspk_core::MemoryRecording;
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

    /// Level-0 buckets of page `page` computed here as a browser helper does: from the samples
    /// alone, `[+∞, −∞]` where no sample falls.
    fn page_rows(rec: &MemoryRecording, p: &Pyramid, page: u64) -> Vec<[f32; 2]> {
        let (start, count) = (page * p.page_samples(), p.page_buckets(page));
        let end = (start + p.page_samples()).min(p.samples());
        let channels: Vec<usize> = (0..rec.info().channel_count()).collect();
        let mut buf = vec![0.0f32; channels.len() * (end - start) as usize];
        rec.read(&channels, start..end, &mut buf).unwrap();
        let mut rows = vec![EMPTY; channels.len() * count];
        for (c, row) in buf.chunks_exact((end - start) as usize).enumerate() {
            for (i, &v) in row.iter().enumerate() {
                let b = &mut rows[c * count + i / p.base() as usize];
                *b = [b[0].min(v), b[1].max(v)];
            }
        }
        rows
    }

    #[test]
    fn pages_from_buckets_equal_pages_filled_in_any_order() {
        let rec = recording(3, 100_003);
        let (filled, given) = (Pyramid::in_memory(&rec, 16).unwrap(), Pyramid::in_memory(&rec, 16).unwrap());
        filled.fill(&rec, 0, 100_003, 1 << 16, || false).unwrap();
        // Last page first, then the middle outwards: the order a focus gives
        let pages = given.pages();
        let mut order: Vec<u64> = (0..pages).collect();
        order.sort_by_key(|&q| (q as i64 - pages as i64 / 2).abs());
        order.insert(0, pages - 1);
        for q in order {
            let rows = page_rows(&rec, &given, q);
            given.build_page_from_buckets(q, &rows).unwrap();
        }
        assert!(given.is_complete());
        assert!(given.built_columns(7).iter().all(|&b| b));
        for &(start, end, width) in &[(0u64, 100_003u64, 64usize), (12_345, 54_321, 33), (99_000, 100_003, 4)] {
            let (mut a, mut b) = (vec![[0.0; 2]; 3 * width], vec![[0.0; 2]; 3 * width]);
            filled.envelope(&[0, 1, 2], start, end, width, &mut a).unwrap();
            given.envelope(&[0, 1, 2], start, end, width, &mut b).unwrap();
            assert_eq!(a, b, "{start}..{end} at {width}");
        }
        // A page given twice, or with the wrong length, changes nothing
        assert!(!given.build_page_from_buckets(0, &page_rows(&rec, &given, 0)).unwrap());
        assert!(Pyramid::in_memory(&rec, 16).unwrap().build_page_from_buckets(0, &[EMPTY; 3]).is_err());
    }

    #[test]
    fn a_complete_pyramid_restores_from_its_bytes() {
        let rec = recording(2, 40_000);
        let p = Pyramid::in_memory(&rec, 16).unwrap();
        assert!(p.complete_bytes().is_none(), "not before it is complete");
        p.fill(&rec, 0, 40_000, 1 << 16, || false).unwrap();
        let bytes = p.complete_bytes().unwrap();
        let back = Pyramid::in_memory(&rec, 16).unwrap();
        assert!(!back.restore(&bytes[1..]), "a wrong length is refused");
        assert!(back.restore(&bytes));
        assert!(back.is_complete());
        let (mut a, mut b) = (vec![[0.0; 2]; 2 * 50], vec![[0.0; 2]; 2 * 50]);
        p.envelope(&[0, 1], 0, 40_000, 50, &mut a).unwrap();
        back.envelope(&[0, 1], 0, 40_000, 50, &mut b).unwrap();
        assert_eq!(a, b);
        // A pyramid that started building is not overwritten
        let started = Pyramid::in_memory(&rec, 16).unwrap();
        started.fill(&rec, 0, 100, 1 << 16, || false).unwrap();
        assert!(!started.restore(&bytes));
    }

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("dk_pyramid_{name}_{}", std::process::id()))
    }

    fn identity(rec: &MemoryRecording) -> PyramidIdentity {
        let info = rec.info();
        PyramidIdentity { source_id: "main".into(), len: 0, mtime_ns: 0, channels: info.channel_count(), samples: info.samples, sample_rate_hz: info.sample_rate_hz(), signature: [0u8; 32] }
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
    fn files_resume_partial_builds_and_invalidate_on_change() {
        let rec = recording(2, 50_000);
        let dir = temp_dir("reopen");
        let path = pyramid_path(&dir.join("rec.bin"), "main");
        let id = identity(&rec);

        // Partial build: persists ready pages and resumes on reopen without starting over
        let p = Pyramid::create_file(&path, &id, 8).unwrap();
        assert!(p.fill(&rec, 0, 20_000, 1 << 20, || false).unwrap());
        let done_before = p.done_samples();
        assert!(done_before >= 20_000 && done_before < 50_000);
        drop(p);

        let resumed = Pyramid::open_file(&path, &id, 8).unwrap().expect("partial files resume");
        assert!(!resumed.is_complete());
        assert_eq!(resumed.done_samples(), done_before);
        assert!(resumed.covers(0, 20_000));
        // Finish the rest
        assert!(resumed.fill(&rec, 0, 50_000, 1 << 20, || false).unwrap());
        assert!(resumed.is_complete());
        drop(resumed);

        let reopened = Pyramid::open_file(&path, &id, 8).unwrap().expect("complete files reopen");
        assert!(reopened.is_complete() && reopened.covers(0, 50_000));
        let mut out = vec![[0.0; 2]; 50];
        assert!(reopened.envelope(&[1], 0, 50_000, 50, &mut out).unwrap());
        assert!(out.iter().all(|v| v[0].is_finite()));
        drop(reopened);
        assert!(Pyramid::open_file(&path, &PyramidIdentity { mtime_ns: 1, ..id.clone() }, 8).unwrap().is_none());
        assert!(Pyramid::open_file(&path, &id.clone().with_signature([7u8; 32]), 8).unwrap().is_none());
        assert!(Pyramid::open_file(&path, &id, 16).unwrap().is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn clamped_envelope_falls_back_to_level_0_for_fine_windows() {
        let rec = recording(2, 4_000);
        let p = Pyramid::in_memory(&rec, 64).unwrap();
        assert!(p.fill(&rec, 0, 4_000, 1 << 16, || false).unwrap());
        let mut out = vec![[0.0; 2]; 2 * 100];
        // 1000 samples / 100 cols = 10 samples/col (< base 64): normal envelope returns false, clamped reads level 0.
        assert!(!p.envelope(&[0, 1], 0, 1_000, 100, &mut out).unwrap());
        p.envelope_clamped(&[0, 1], 0, 1_000, 100, &mut out).unwrap();
        assert!(out.iter().all(|v| v[0].is_finite() && v[1].is_finite() && v[0] <= v[1]));
    }
}
