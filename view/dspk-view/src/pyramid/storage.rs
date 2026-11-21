//! The bytes behind a pyramid: an anonymous mapping (in memory; the OS commits only the pages
//! that are filled) or a file next to the recording (kept between sessions).
//!
//! Files live in `<dir>/cache/<file name>.cache/`:
//! - `raw/<rec_uuid>_<source>.minmax` ([`pyramid_path`]): the raw recording's min/max pyramid;
//! - `drv/<drv_uuid>_drv_<source>.minmax` ([`drv_pyramid_path`]): a derived chain's min/max pyramid.
//!
//! Layout on disk (version 2):
//! - `[0 .. HEADER_BYTES)`: header (magic, version, complete flag, identity, 32-byte chain signature,
//!   source id);
//! - `[HEADER_BYTES .. data_offset)`: persisted page-ready bitmap (one bit per page, padded to a
//!   4096-byte page boundary), so partial builds resume across restarts instead of starting over;
//! - `[data_offset .. bytes)`: level data, page-aligned.

use std::fs::{self, OpenOptions};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use dspk_core::{DspError, DspResult, RecordingSource};
use memmap2::{MmapMut, MmapRaw};

const MAGIC: &[u8; 8] = b"DKMINMAX";
const VERSION: u32 = 2;
/// Byte offset of the complete flag in the header.
const COMPLETE_AT: usize = 12;
/// Header bytes of a file; bitmap starts page-aligned after it.
pub const HEADER_BYTES: usize = 4096;
const MAX_ID_BYTES: usize = 1024;

/// Generates a random UUIDv4 string (`xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx`).
pub fn new_uuid() -> String {
    let mut b = [0u8; 16];
    let _ = getrandom::fill(&mut b);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    )
}

fn sanitize(s: &str, fallback: &str) -> String {
    let id: String = s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    if id.is_empty() { fallback.to_string() } else { id }
}

/// `<dir>/cache/<file name>.cache` for the recording at `path`.
pub fn cache_dir(path: &Path) -> PathBuf {
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path.file_name().map_or_else(|| "recording".into(), |n| n.to_string_lossy().into_owned());
    dir.join("cache").join(format!("{name}.cache"))
}

/// Persistent UUID tag for the recording at `path`, stored in `<dir>/cache/<file name>.cache/id`.
/// Created once on first open and reused across restarts.
pub fn recording_uuid(path: &Path) -> DspResult<String> {
    let dir = cache_dir(path);
    let id_file = dir.join("id");
    if let Ok(existing) = fs::read_to_string(&id_file) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    fs::create_dir_all(&dir).map_err(|e| DspError::Io(format!("{}: {e}", dir.display())))?;
    let id = new_uuid();
    // Create atomically if another thread/process didn't beat us to it
    match OpenOptions::new().write(true).create_new(true).open(&id_file) {
        Ok(mut f) => {
            use std::io::Write;
            f.write_all(id.as_bytes()).map_err(|e| DspError::Io(format!("{}: {e}", id_file.display())))?;
            Ok(id)
        }
        Err(_) => {
            let existing = fs::read_to_string(&id_file).map_err(|e| DspError::Io(format!("{}: {e}", id_file.display())))?;
            let trimmed = existing.trim();
            if trimmed.is_empty() { Ok(id) } else { Ok(trimmed.to_string()) }
        }
    }
}

/// `<dir>/cache/<file name>.cache/raw/<rec_uuid>_<source id>.minmax` for the recording at `path`.
pub fn pyramid_path(path: &Path, source_id: &str) -> PathBuf {
    let rec_id = recording_uuid(path).unwrap_or_else(|_| "raw".to_string());
    let id = sanitize(source_id, "main");
    cache_dir(path).join("raw").join(format!("{rec_id}_{id}.minmax"))
}

/// `<dir>/cache/<file name>.cache/drv/<drv_uuid>_drv_<source id>.minmax` for a derived chain on
/// `source_id` of the recording at `path`.
pub fn drv_pyramid_path(path: &Path, source_id: &str, drv_id: &str) -> PathBuf {
    let id = sanitize(source_id, "main");
    let drv = sanitize(drv_id, "drv");
    cache_dir(path).join("drv").join(format!("{drv}_drv_{id}.minmax"))
}

/// Page-aligned byte size of the persisted ready-page bitmap for `pages` pages.
pub(crate) fn bitmap_bytes(pages: u64) -> usize {
    let words = pages.div_ceil(64) as usize;
    let raw = words * std::mem::size_of::<u64>();
    raw.div_ceil(HEADER_BYTES) * HEADER_BYTES
}

/// Byte offset where level-0 data begins in a file with `pages` pages.
pub(crate) fn data_offset(pages: u64) -> usize {
    HEADER_BYTES + bitmap_bytes(pages)
}

/// What a pyramid file was built from; a mismatch means the recording (or derived chain) changed
/// and the file is rebuilt in place.
#[derive(Debug, Clone, PartialEq)]
pub struct PyramidIdentity {
    /// Source id within the recording (several signals can share one file).
    pub source_id: String,
    /// Size in bytes of the recording file (0 for folders).
    pub len: u64,
    /// Modification time of the recording (nanoseconds since the Unix epoch).
    pub mtime_ns: u64,
    pub channels: usize,
    pub samples: u64,
    pub sample_rate_hz: f64,
    /// Chain signature (all zeros for raw; SHA-256 of canonical ops + channels for derived chains).
    pub signature: [u8; 32],
}

impl PyramidIdentity {
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
            signature: [0u8; 32],
        })
    }

    /// The same identity tagged with a derived-chain signature.
    pub fn with_signature(mut self, signature: [u8; 32]) -> Self {
        self.signature = signature;
        self
    }

    /// Header with the complete flag cleared: magic, version, flag, identity, signature, source id.
    fn header(&self, base: u64) -> Vec<u8> {
        let mut h = Vec::with_capacity(HEADER_BYTES);
        h.extend_from_slice(MAGIC);
        h.extend_from_slice(&VERSION.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        for v in [self.channels as u64, self.samples, base, self.len, self.mtime_ns, self.sample_rate_hz.to_bits()] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        h.extend_from_slice(&self.signature);
        h.extend_from_slice(&(self.source_id.len() as u32).to_le_bytes());
        h.extend_from_slice(self.source_id.as_bytes());
        h.resize(HEADER_BYTES, 0);
        h
    }
}

pub(crate) struct Storage {
    /// The mapping; `None` for memory taken from the heap ([`Self::heap`]).
    map: Option<ManuallyDrop<MmapRaw>>,
    /// Zeroed heap memory and its length, where there are no anonymous mappings (a browser).
    heap: Option<(*mut u8, usize)>,
    /// The file, for file-backed storage.
    path: Option<PathBuf>,
    /// Length of the persisted ready-page bitmap in bytes (0 for in-memory storage).
    bitmap_bytes: usize,
}

impl Storage {
    /// `bytes` of zeroed anonymous memory.
    #[cfg(not(target_family = "wasm"))]
    pub fn memory(bytes: usize) -> DspResult<Self> {
        let map = MmapMut::map_anon(bytes.max(1))?;
        Ok(Self { map: Some(ManuallyDrop::new(MmapRaw::from(map))), heap: None, path: None, bitmap_bytes: 0 })
    }

    /// `bytes` of zeroed memory from the heap: a browser has no anonymous mappings (all of it is
    /// committed at once, unlike a mapping).
    #[cfg(target_family = "wasm")]
    pub fn memory(bytes: usize) -> DspResult<Self> {
        let _ = MmapMut::map_anon;
        let boxed = vec![0u8; bytes.max(1)].into_boxed_slice();
        let len = boxed.len();
        Ok(Self { map: None, heap: Some((Box::into_raw(boxed).cast::<u8>(), len)), path: None, bitmap_bytes: 0 })
    }

    /// Opens an existing file at `path` built from `identity` with level-0 bucket `base`, `bytes`
    /// long and `bitmap_words` ready-page words. Returns `(storage, ready_words, complete)` for
    /// both complete and partial builds; `None` when missing, wrong size, older version, or
    /// mismatched identity / chain signature.
    pub fn open_file(path: &Path, identity: &PyramidIdentity, base: u64, bytes: usize, bitmap_words: usize) -> DspResult<Option<(Self, Vec<u64>, bool)>> {
        let Ok(file) = OpenOptions::new().read(true).write(true).open(path) else { return Ok(None) };
        if file.metadata().map(|m| m.len()).unwrap_or(0) != bytes as u64 {
            return Ok(None);
        }
        let map = MmapRaw::map_raw(&file)?;
        // SAFETY: the mapping is `bytes ≥ HEADER_BYTES + bitmap_bytes` long (checked above).
        let header = unsafe { std::slice::from_raw_parts(map.as_ptr(), HEADER_BYTES) };
        let expected = identity.header(base);
        let complete = u32::from_le_bytes(header[COMPLETE_AT..COMPLETE_AT + 4].try_into().expect("4 bytes")) == 1;
        if header[..COMPLETE_AT] != expected[..COMPLETE_AT] || header[COMPLETE_AT + 4..] != expected[COMPLETE_AT + 4..] {
            return Ok(None);
        }
        let bm_bytes = (bitmap_words * std::mem::size_of::<u64>()).div_ceil(HEADER_BYTES) * HEADER_BYTES;
        let mut ready_words = vec![0u64; bitmap_words];
        if !complete && bitmap_words > 0 {
            // SAFETY: `HEADER_BYTES + bitmap_words * 8 <= bytes`.
            let raw = unsafe { std::slice::from_raw_parts(map.as_ptr().add(HEADER_BYTES), bitmap_words * 8) };
            for (w, chunk) in ready_words.iter_mut().zip(raw.chunks_exact(8)) {
                *w = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
            }
        }
        Ok(Some((Self { map: Some(ManuallyDrop::new(map)), heap: None, path: Some(path.to_path_buf()), bitmap_bytes: bm_bytes }, ready_words, complete)))
    }

    /// A new, empty file at `path` (replacing any other in place), `bytes` long, headed by
    /// `identity` with a zeroed ready-page bitmap.
    pub fn create_file(path: &Path, identity: &PyramidIdentity, base: u64, bytes: usize, bitmap_words: usize) -> DspResult<Self> {
        if identity.source_id.len() > MAX_ID_BYTES {
            return Err(DspError::InvalidConfig(format!("source id longer than {MAX_ID_BYTES} bytes")));
        }
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).map_err(|e| DspError::Io(format!("{}: {e}", dir.display())))?;
        }
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)
            .map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
        file.set_len(bytes as u64)?;
        let map = MmapRaw::map_raw(&file)?;
        let header = identity.header(base);
        // SAFETY: the mapping is `bytes ≥ HEADER_BYTES` long and nothing else uses it yet.
        unsafe { std::ptr::copy_nonoverlapping(header.as_ptr(), map.as_mut_ptr(), HEADER_BYTES) };
        let bm_bytes = (bitmap_words * std::mem::size_of::<u64>()).div_ceil(HEADER_BYTES) * HEADER_BYTES;
        Ok(Self { map: Some(ManuallyDrop::new(map)), heap: None, path: Some(path.to_path_buf()), bitmap_bytes: bm_bytes })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn ptr(&self) -> *mut u8 {
        match (&self.map, self.heap) {
            (Some(map), _) => map.as_mut_ptr(),
            (None, Some((ptr, _))) => ptr,
            (None, None) => unreachable!("storage is a mapping or heap memory"),
        }
    }

    /// Persists one `u64` word of the ready-page bitmap into the mmapped file (no-op in memory).
    pub fn write_ready_word(&self, word_idx: usize, value: u64) {
        let Some(map) = self.map.as_ref().filter(|_| self.path.is_some() && self.bitmap_bytes > 0) else { return };
        let offset = HEADER_BYTES + word_idx * std::mem::size_of::<u64>();
        if offset + 8 <= HEADER_BYTES + self.bitmap_bytes {
            // SAFETY: within the mmapped bitmap region; callers pass the atomically updated word.
            unsafe { map.as_mut_ptr().add(offset).cast::<[u8; 8]>().write_unaligned(value.to_le_bytes()) };
        }
    }

    /// Flushes a complete file and only then sets its complete flag (nothing for memory).
    pub fn mark_complete(&self) -> DspResult<()> {
        let Some(map) = self.map.as_ref().filter(|_| self.path.is_some()) else { return Ok(()) };
        map.flush()?;
        // SAFETY: the flag lies in the header, which only this call writes after creation.
        unsafe { map.as_mut_ptr().add(COMPLETE_AT).cast::<[u8; 4]>().write_unaligned(1u32.to_le_bytes()) };
        map.flush_range(0, HEADER_BYTES)?;
        Ok(())
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if let Some(map) = self.map.as_mut() {
            // Persist any ready-page bitmap updates on clean shutdown so partial builds resume
            if self.path.is_some() && self.bitmap_bytes > 0 {
                let _ = map.flush_range(HEADER_BYTES, self.bitmap_bytes);
            }
            // SAFETY: not used after this point.
            unsafe { ManuallyDrop::drop(map) };
        }
        if let Some((ptr, len)) = self.heap.take() {
            // SAFETY: made by `Box::into_raw` of a `[u8]` of this length in `memory`, freed once.
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
        }
    }
}

// SAFETY: the heap memory is owned by the storage and used exactly as the mapping is (callers
// write disjoint ranges); `MmapRaw` itself is Send and Sync.
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_paths_use_rec_cache_raw_and_drv_folders() {
        let dir = std::env::temp_dir().join(format!("dk_storage_paths_{}", std::process::id()));
        let rec = dir.join("a.ap.bin");
        let raw = pyramid_path(&rec, "ElectricalSeries/x");
        let uuid = recording_uuid(&rec).unwrap();
        assert_eq!(raw, dir.join("cache/a.ap.bin.cache/raw").join(format!("{uuid}_ElectricalSeries_x.minmax")));
        assert_eq!(pyramid_path(&rec, "ElectricalSeries/x"), raw, "recording uuid is stable across opens");
        let drv = drv_pyramid_path(&rec, "ElectricalSeries/x", "abc-123");
        assert_eq!(drv, dir.join("cache/a.ap.bin.cache/drv/abc-123_drv_ElectricalSeries_x.minmax"));
        let _ = fs::remove_dir_all(&dir);
    }
}
