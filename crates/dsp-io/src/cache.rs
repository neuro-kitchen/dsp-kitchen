//! Multi-resolution min/max cache of a recording, for drawing zoomed-out views exactly.
//!
//! Level 0 holds the `[min, max]` of every `base` consecutive samples per channel; each next level
//! halves the bucket count, down to one bucket for the whole recording. An envelope is drawn from
//! the finest level with at least one bucket per pixel column, with column edges aligned to that
//! level's buckets: every sample falls into exactly one column, so no peak is lost (column edges
//! move by less than one column). Windows finer than `base` samples per column are read raw.
//!
//! `base` only trades cache size against how far one zooms out before the cache takes over; the
//! drawn envelope is exact for any value. The cache holds `≈ 4 / base` floats per sample (all
//! levels), so [`DEFAULT_BASE`] = 64 costs 1/16 of the recording as f32.
//!
//! The file lives next to the recording (`<dir>/cache/<file name>/<source>.minmax`, see
//! [`cache_path`]) and is valid while the recording's size, modification time, shape and rate
//! match its header. Building is chunked and cancellable; levels are readable while they fill.

use std::fs::{self, OpenOptions};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use dsp_core::{DspError, DspResult, RecordingSource};
use memmap2::MmapRaw;

const MAGIC: &[u8; 8] = b"DKMINMAX";
const VERSION: u32 = 1;
/// Header bytes; level data starts page-aligned after it.
const HEADER_BYTES: usize = 4096;
const MAX_ID_BYTES: usize = 1024;

/// Default level-0 bucket (samples); see the module docs for what it trades.
pub const DEFAULT_BASE: u64 = 64;

/// What a cache file was built from; a mismatch means the recording changed and the cache is
/// rebuilt.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheIdentity {
    /// Source id within the recording (several signals can share one file).
    pub source_id: String,
    /// Size in bytes of the recording file (0 for folders).
    pub len: u64,
    /// Modification time of the recording (nanoseconds since the Unix epoch).
    pub mtime_ns: u64,
    pub channels: usize,
    pub samples: u64,
    pub sample_rate_hz: f64,
}

impl CacheIdentity {
    /// Identity of source `source_id` of the recording at `path` with `source`'s shape. For a
    /// folder store the modification time of its `zarr.json` is used when present.
    pub fn of(path: &Path, source_id: &str, source: &dyn RecordingSource) -> DspResult<Self> {
        let stamp = if path.is_dir() && path.join("zarr.json").exists() { path.join("zarr.json") } else { path.to_path_buf() };
        let meta = fs::metadata(&stamp).map_err(|e| DspError::Io(format!("{}: {e}", stamp.display())))?;
        let mtime_ns = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as u64);
        let info = source.info();
        Ok(Self {
            source_id: source_id.to_string(),
            len: if path.is_dir() { 0 } else { meta.len() },
            mtime_ns,
            channels: info.channel_count(),
            samples: info.samples,
            sample_rate_hz: info.sample_rate_hz(),
        })
    }

    /// Identity of an in-memory or procedural source (only used for temporary caches).
    pub fn transient(source: &dyn RecordingSource) -> Self {
        let info = source.info();
        Self {
            source_id: info.name.clone(),
            len: 0,
            mtime_ns: 0,
            channels: info.channel_count(),
            samples: info.samples,
            sample_rate_hz: info.sample_rate_hz(),
        }
    }
}

/// `<dir>/cache/<file name>/<source id>.minmax` for the recording at `path`.
pub fn cache_path(path: &Path, source_id: &str) -> PathBuf {
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path.file_name().map_or_else(|| "recording".into(), |n| n.to_string_lossy().into_owned());
    let id: String = source_id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    let id = if id.is_empty() { "main".to_string() } else { id };
    dir.join("cache").join(name).join(format!("{id}.minmax"))
}

#[derive(Debug, Clone, Copy)]
struct Level {
    /// Samples per bucket.
    bucket: u64,
    buckets: u64,
    /// Byte offset of the level's `[channel][bucket][min, max]` f32 data.
    offset: usize,
}

fn level_layout(channels: usize, samples: u64, base: u64) -> (Vec<Level>, usize) {
    let mut levels = Vec::new();
    let mut offset = HEADER_BYTES;
    let (mut bucket, mut buckets) = (base, samples.div_ceil(base).max(1));
    loop {
        levels.push(Level { bucket, buckets, offset });
        offset += channels * buckets as usize * 8;
        if buckets == 1 {
            break;
        }
        bucket *= 2;
        buckets = buckets.div_ceil(2);
    }
    (levels, offset)
}

/// A min/max pyramid backed by a memory-mapped file. Shareable across threads: one thread
/// [`build`](Self::build)s while others read [`envelope`](Self::envelope)s of the ready part.
pub struct MinMaxCache {
    map: ManuallyDrop<MmapRaw>,
    path: PathBuf,
    temporary: bool,
    channels: usize,
    samples: u64,
    base: u64,
    levels: Vec<Level>,
    /// Buckets written per level (a prefix).
    ready: Vec<AtomicU64>,
}

// SAFETY: the mapping is only written by `build` in bucket ranges not yet published through
// `ready` (Release), and only read in ranges already published (Acquire).
unsafe impl Send for MinMaxCache {}
unsafe impl Sync for MinMaxCache {}

impl MinMaxCache {
    /// Opens a complete cache at `path` built from `identity` with level-0 bucket `base`;
    /// `None` when it is missing, incomplete, of another version or built from something else.
    pub fn open(path: &Path, identity: &CacheIdentity, base: u64) -> DspResult<Option<Self>> {
        let Ok(file) = OpenOptions::new().read(true).write(true).open(path) else { return Ok(None) };
        let (levels, total) = level_layout(identity.channels, identity.samples, base);
        if file.metadata().map(|m| m.len()).unwrap_or(0) != total as u64 {
            return Ok(None);
        }
        let map = MmapRaw::map_raw(&file)?;
        let header = unsafe { std::slice::from_raw_parts(map.as_ptr(), HEADER_BYTES) };
        if header[..8] != MAGIC[..] || header_u32(header, 8) != VERSION || header_u32(header, 12) != 1 || header[16..HEADER_BYTES] != encode_identity(identity, base)[16..] {
            return Ok(None);
        }
        let ready = levels.iter().map(|l| AtomicU64::new(l.buckets)).collect();
        Ok(Some(Self { map: ManuallyDrop::new(map), path: path.to_path_buf(), temporary: false, channels: identity.channels, samples: identity.samples, base, levels, ready }))
    }

    /// Creates an empty cache file at `path` (replacing any other); fill it with
    /// [`build`](Self::build).
    pub fn create(path: &Path, identity: &CacheIdentity, base: u64) -> DspResult<Self> {
        Self::create_file(path, identity, base, false)
    }

    /// An empty cache in the system temporary folder, deleted when dropped (for sources that are
    /// not files).
    pub fn temporary(identity: &CacheIdentity, base: u64) -> DspResult<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = format!("dsp-kitchen-{}-{}.minmax", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
        Self::create_file(&std::env::temp_dir().join(name), identity, base, true)
    }

    fn create_file(path: &Path, identity: &CacheIdentity, base: u64, temporary: bool) -> DspResult<Self> {
        if base == 0 {
            return Err(DspError::InvalidConfig("min/max cache base must be at least 1 sample".into()));
        }
        if identity.source_id.len() > MAX_ID_BYTES {
            return Err(DspError::InvalidConfig(format!("source id longer than {MAX_ID_BYTES} bytes")));
        }
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).map_err(|e| DspError::Io(format!("{}: {e}", dir.display())))?;
        }
        let (levels, total) = level_layout(identity.channels, identity.samples, base);
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)
            .map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
        file.set_len(total as u64)?;
        let map = MmapRaw::map_raw(&file)?;
        let header = encode_identity(identity, base);
        unsafe { std::ptr::copy_nonoverlapping(header.as_ptr(), map.as_mut_ptr(), HEADER_BYTES) };
        let ready = levels.iter().map(|_| AtomicU64::new(0)).collect();
        Ok(Self { map: ManuallyDrop::new(map), path: path.to_path_buf(), temporary, channels: identity.channels, samples: identity.samples, base, levels, ready })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    /// Samples from the start whose level-0 buckets are built.
    pub fn ready_samples(&self) -> u64 {
        (self.ready[0].load(Ordering::Acquire) * self.base).min(self.samples)
    }

    pub fn is_complete(&self) -> bool {
        self.levels.iter().zip(&self.ready).all(|(l, r)| r.load(Ordering::Acquire) == l.buckets)
    }

    fn slot(&self, level: usize, channel: usize, bucket: u64) -> *mut [f32; 2] {
        let l = &self.levels[level];
        let index = channel * l.buckets as usize + bucket as usize;
        unsafe { self.map.as_mut_ptr().add(l.offset + index * 8) as *mut [f32; 2] }
    }

    fn get(&self, level: usize, channel: usize, bucket: u64) -> [f32; 2] {
        unsafe { self.slot(level, channel, bucket).read_unaligned() }
    }

    /// Fills the cache from `source` (which must match the identity it was created with),
    /// reading all channels `chunk_samples` at a time (rounded up to whole level-0 buckets).
    /// Returns `false` when `cancel` stopped it; the file then stays incomplete and is rebuilt on
    /// the next [`open`](Self::open). Build a cache once: a second call starts over.
    pub fn build(&self, source: &dyn RecordingSource, chunk_samples: u64, cancel: &AtomicBool, mut progress: impl FnMut(u64, u64)) -> DspResult<bool> {
        let info = source.info();
        if info.channel_count() != self.channels || info.samples != self.samples {
            return Err(DspError::InvalidConfig("source does not match the min/max cache shape".into()));
        }
        for r in &self.ready {
            r.store(0, Ordering::Release);
        }
        let chunk = chunk_samples.max(1).div_ceil(self.base) * self.base;
        let channels: Vec<usize> = (0..self.channels).collect();
        let mut buf = Vec::new();
        let mut s0 = 0u64;
        while s0 < self.samples {
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            let s1 = (s0 + chunk).min(self.samples);
            let n = (s1 - s0) as usize;
            buf.resize(self.channels * n, 0.0);
            source.read(&channels, s0..s1, &mut buf)?;
            let (b0, b1) = (s0 / self.base, s1.div_ceil(self.base));
            for (ch, row) in buf.chunks_exact(n).enumerate() {
                for (b, samples) in (b0..b1).zip(row.chunks(self.base as usize)) {
                    let mm = samples.iter().fold([f32::INFINITY, f32::NEG_INFINITY], |[lo, hi], &v| [lo.min(v), hi.max(v)]);
                    unsafe { self.slot(0, ch, b).write_unaligned(mm) };
                }
            }
            self.ready[0].store(b1, Ordering::Release);
            self.propagate();
            progress(s1, self.samples);
            s0 = s1;
        }
        self.map.flush()?;
        // Mark complete only after the data is on disk
        unsafe { self.map.as_mut_ptr().add(12).cast::<[u8; 4]>().write_unaligned(1u32.to_le_bytes()) };
        self.map.flush_range(0, HEADER_BYTES)?;
        Ok(true)
    }

    /// Builds every coarser bucket whose two finer buckets are ready.
    fn propagate(&self) {
        for k in 1..self.levels.len() {
            let (below, level) = (self.levels[k - 1], self.levels[k]);
            let ready_below = self.ready[k - 1].load(Ordering::Acquire);
            let limit = if ready_below == below.buckets { level.buckets } else { ready_below / 2 };
            let from = self.ready[k].load(Ordering::Acquire);
            for ch in 0..self.channels {
                for b in from..limit {
                    let a = self.get(k - 1, ch, 2 * b);
                    let mm = if 2 * b + 1 < below.buckets {
                        let c = self.get(k - 1, ch, 2 * b + 1);
                        [a[0].min(c[0]), a[1].max(c[1])]
                    } else {
                        a
                    };
                    unsafe { self.slot(k, ch, b).write_unaligned(mm) };
                }
            }
            self.ready[k].store(limit.max(from), Ordering::Release);
        }
    }

    /// `[min, max]` per pixel column of samples `start..end` for each of `channels`, written
    /// row by row into `out` (`channels.len() × width`). Uses the finest level with at least one
    /// bucket per column, column edges aligned to its buckets; columns not built yet are NaN.
    /// Returns `false` (leaving `out` untouched) when a column spans fewer than `base` samples:
    /// read those windows from the recording.
    pub fn envelope(&self, channels: &[usize], start: u64, end: u64, width: usize, out: &mut [[f32; 2]]) -> DspResult<bool> {
        if let Some(&channel) = channels.iter().find(|&&c| c >= self.channels) {
            return Err(DspError::InvalidChannel { channel, total: self.channels });
        }
        if start >= end || end > self.samples || width == 0 {
            return Err(DspError::SampleRange { start, end, total: self.samples });
        }
        if out.len() != channels.len() * width {
            return Err(DspError::ShapeMismatch { expected: vec![channels.len(), width], actual: vec![out.len()] });
        }
        let n = end - start;
        if n < self.base * width as u64 {
            return Ok(false);
        }
        let k = self.levels.iter().rposition(|l| l.bucket * width as u64 <= n).unwrap_or(0);
        let level = self.levels[k];
        let ready = self.ready[k].load(Ordering::Acquire);
        let edge = |x: usize| (start + x as u64 * n / width as u64) / level.bucket;
        for x in 0..width {
            let j0 = edge(x);
            let j1 = if x + 1 == width { end.div_ceil(level.bucket) } else { edge(x + 1) };
            for (r, &ch) in channels.iter().enumerate() {
                out[r * width + x] = if j1 > ready {
                    [f32::NAN, f32::NAN]
                } else {
                    (j0..j1).fold([f32::INFINITY, f32::NEG_INFINITY], |[lo, hi], j| {
                        let [a, b] = self.get(k, ch, j);
                        [lo.min(a), hi.max(b)]
                    })
                };
            }
        }
        Ok(true)
    }

    /// Sample range column `x` of a `width`-column envelope of `start..end` covers (the
    /// bucket-aligned edges [`envelope`](Self::envelope) uses), or `None` for raw windows.
    pub fn column_samples(&self, start: u64, end: u64, width: usize, x: usize) -> Option<(u64, u64)> {
        let n = end.checked_sub(start)?;
        if width == 0 || n < self.base * width as u64 {
            return None;
        }
        let k = self.levels.iter().rposition(|l| l.bucket * width as u64 <= n).unwrap_or(0);
        let bucket = self.levels[k].bucket;
        let edge = |x: usize| (start + x as u64 * n / width as u64) / bucket;
        let j1 = if x + 1 == width { end.div_ceil(bucket) } else { edge(x + 1) };
        Some((edge(x) * bucket, (j1 * bucket).min(self.samples)))
    }
}

impl Drop for MinMaxCache {
    fn drop(&mut self) {
        // SAFETY: not used after this point.
        unsafe { ManuallyDrop::drop(&mut self.map) };
        if self.temporary {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn header_u32(header: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(header[at..at + 4].try_into().expect("4 bytes"))
}

/// Header with `complete = 0`: magic, version, complete flag, then the identity and layout.
fn encode_identity(id: &CacheIdentity, base: u64) -> Vec<u8> {
    let mut h = Vec::with_capacity(HEADER_BYTES);
    h.extend_from_slice(MAGIC);
    h.extend_from_slice(&VERSION.to_le_bytes());
    h.extend_from_slice(&0u32.to_le_bytes());
    for v in [id.channels as u64, id.samples, base, id.len, id.mtime_ns, id.sample_rate_hz.to_bits()] {
        h.extend_from_slice(&v.to_le_bytes());
    }
    h.extend_from_slice(&(id.source_id.len() as u32).to_le_bytes());
    h.extend_from_slice(id.source_id.as_bytes());
    h.resize(HEADER_BYTES, 0);
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_core::MemoryRecording;

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
    fn expected(rec: &MemoryRecording, cache: &MinMaxCache, channels: &[usize], start: u64, end: u64, width: usize) -> Vec<[f32; 2]> {
        let mut out = vec![[0.0; 2]; channels.len() * width];
        for x in 0..width {
            let (c0, c1) = cache.column_samples(start, end, width, x).unwrap();
            let mut buf = vec![0.0f32; channels.len() * (c1 - c0) as usize];
            rec.read(channels, c0..c1, &mut buf).unwrap();
            for (r, row) in buf.chunks_exact((c1 - c0) as usize).enumerate() {
                out[r * width + x] = row.iter().fold([f32::INFINITY, f32::NEG_INFINITY], |[a, b], &v| [a.min(v), b.max(v)]);
            }
        }
        out
    }

    #[test]
    fn envelope_matches_brute_force_and_keeps_every_peak() {
        let rec = recording(5, 100_003);
        let cache = MinMaxCache::temporary(&CacheIdentity::transient(&rec), 16).unwrap();
        assert!(cache.build(&rec, 1_000, &AtomicBool::new(false), |_, _| {}).unwrap());
        assert!(cache.is_complete());

        let channels = [4, 0, 2];
        for (start, end, width) in [(0u64, 100_003u64, 97usize), (123, 99_000, 640), (5_000, 7_000, 100), (0, 100_003, 1)] {
            let mut out = vec![[0.0; 2]; channels.len() * width];
            assert!(cache.envelope(&channels, start, end, width, &mut out).unwrap());
            assert_eq!(out, expected(&rec, &cache, &channels, start, end, width), "{start}..{end} @ {width}");
            // Every spike inside the window shows up in some column
            let spikes = (start..end).filter(|s| (*s as usize + 37 * 4) % 4099 == 0).count();
            if spikes > 0 {
                assert!(out[..width].iter().any(|v| v[0] < -8_000.0));
            }
        }
        // Finer than one bucket per column: read raw
        let mut out = vec![[0.0; 2]; 3 * 100];
        assert!(!cache.envelope(&channels, 0, 1_000, 100, &mut out).unwrap());
    }

    #[test]
    fn partial_build_marks_unbuilt_columns_and_reopen_checks_identity() {
        let rec = recording(2, 50_000);
        let dir = std::env::temp_dir().join(format!("dk_cache_{}", std::process::id()));
        let path = cache_path(&dir.join("rec.bin"), "main");
        let id = CacheIdentity::transient(&rec);

        let cancel = AtomicBool::new(false);
        let cache = MinMaxCache::create(&path, &id, 8).unwrap();
        cache.build(&rec, 10_000, &cancel, |done, _| cancel.store(done >= 20_000, Ordering::Relaxed)).unwrap();
        assert_eq!(cache.ready_samples(), 20_000);
        assert!(!cache.is_complete());
        let mut out = vec![[0.0; 2]; 2 * 50];
        assert!(cache.envelope(&[0, 1], 0, 50_000, 50, &mut out).unwrap());
        let built = out[..50].iter().take_while(|v| v[0].is_finite()).count();
        assert!(built > 0 && built < 50 && out[built..50].iter().all(|v| v[0].is_nan()));
        drop(cache);
        assert!(MinMaxCache::open(&path, &id, 8).unwrap().is_none(), "incomplete caches are rebuilt");

        let cache = MinMaxCache::create(&path, &id, 8).unwrap();
        assert!(cache.build(&rec, 10_000, &AtomicBool::new(false), |_, _| {}).unwrap());
        drop(cache);
        let reopened = MinMaxCache::open(&path, &id, 8).unwrap().expect("complete cache reopens");
        assert!(reopened.is_complete());
        assert!(reopened.envelope(&[1], 0, 50_000, 50, &mut out[..50]).unwrap());
        drop(reopened);
        assert!(MinMaxCache::open(&path, &CacheIdentity { mtime_ns: 1, ..id.clone() }, 8).unwrap().is_none());
        assert!(MinMaxCache::open(&path, &id, 16).unwrap().is_none());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cache_path_sits_next_to_the_recording() {
        assert_eq!(cache_path(Path::new("/data/a.ap.bin"), "ElectricalSeries/x"), PathBuf::from("/data/cache/a.ap.bin/ElectricalSeries_x.minmax"));
    }
}
