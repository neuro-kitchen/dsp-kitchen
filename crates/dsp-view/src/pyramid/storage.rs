//! The bytes behind a pyramid: an anonymous mapping (in memory; the OS commits only the pages
//! that are filled) or a file next to the recording (kept between sessions).
//!
//! A file is `<dir>/cache/<file name>/<source>.minmax` (see [`pyramid_path`]): a
//! [`HEADER_BYTES`] header (magic, version, a complete flag, the [`PyramidIdentity`] and `base`)
//! then the levels. It is valid only when complete and built from the same identity.

use std::fs::{self, OpenOptions};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use dsp_core::{DspError, DspResult, RecordingSource};
use memmap2::{MmapMut, MmapRaw};

const MAGIC: &[u8; 8] = b"DKMINMAX";
const VERSION: u32 = 1;
/// Byte offset of the complete flag in the header.
const COMPLETE_AT: usize = 12;
/// Header bytes of a file; level data starts page-aligned after it.
pub const HEADER_BYTES: usize = 4096;
const MAX_ID_BYTES: usize = 1024;

/// What a pyramid file was built from; a mismatch means the recording changed and the file is
/// rebuilt.
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
        })
    }

    /// Header with the complete flag cleared: magic, version, flag, then identity and `base`.
    fn header(&self, base: u64) -> Vec<u8> {
        let mut h = Vec::with_capacity(HEADER_BYTES);
        h.extend_from_slice(MAGIC);
        h.extend_from_slice(&VERSION.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        for v in [self.channels as u64, self.samples, base, self.len, self.mtime_ns, self.sample_rate_hz.to_bits()] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        h.extend_from_slice(&(self.source_id.len() as u32).to_le_bytes());
        h.extend_from_slice(self.source_id.as_bytes());
        h.resize(HEADER_BYTES, 0);
        h
    }
}

/// `<dir>/cache/<file name>/<source id>.minmax` for the recording at `path`.
pub fn pyramid_path(path: &Path, source_id: &str) -> PathBuf {
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path.file_name().map_or_else(|| "recording".into(), |n| n.to_string_lossy().into_owned());
    let id: String = source_id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    let id = if id.is_empty() { "main".to_string() } else { id };
    dir.join("cache").join(name).join(format!("{id}.minmax"))
}

pub(crate) struct Storage {
    map: ManuallyDrop<MmapRaw>,
    /// The file, for file-backed storage.
    path: Option<PathBuf>,
}

impl Storage {
    /// `bytes` of zeroed anonymous memory.
    pub fn memory(bytes: usize) -> DspResult<Self> {
        let map = MmapMut::map_anon(bytes.max(1))?;
        Ok(Self { map: ManuallyDrop::new(MmapRaw::from(map)), path: None })
    }

    /// The complete file at `path` built from `identity` with level-0 bucket `base` and `bytes`
    /// long; `None` when it is missing, incomplete, of another version or built from something
    /// else.
    pub fn open_file(path: &Path, identity: &PyramidIdentity, base: u64, bytes: usize) -> DspResult<Option<Self>> {
        let Ok(file) = OpenOptions::new().read(true).write(true).open(path) else { return Ok(None) };
        if file.metadata().map(|m| m.len()).unwrap_or(0) != bytes as u64 {
            return Ok(None);
        }
        let map = MmapRaw::map_raw(&file)?;
        // SAFETY: the mapping is `bytes ≥ HEADER_BYTES` long (checked above) and not written yet.
        let header = unsafe { std::slice::from_raw_parts(map.as_ptr(), HEADER_BYTES) };
        let expected = identity.header(base);
        let complete = u32::from_le_bytes(header[COMPLETE_AT..COMPLETE_AT + 4].try_into().expect("4 bytes")) == 1;
        if header[..COMPLETE_AT] != expected[..COMPLETE_AT] || !complete || header[COMPLETE_AT + 4..] != expected[COMPLETE_AT + 4..] {
            return Ok(None);
        }
        Ok(Some(Self { map: ManuallyDrop::new(map), path: Some(path.to_path_buf()) }))
    }

    /// A new, empty file at `path` (replacing any other), `bytes` long, headed by `identity`.
    pub fn create_file(path: &Path, identity: &PyramidIdentity, base: u64, bytes: usize) -> DspResult<Self> {
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
        Ok(Self { map: ManuallyDrop::new(map), path: Some(path.to_path_buf()) })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn ptr(&self) -> *mut u8 {
        self.map.as_mut_ptr()
    }

    /// Flushes a complete file and only then sets its complete flag (nothing for memory).
    pub fn mark_complete(&self) -> DspResult<()> {
        if self.path.is_none() {
            return Ok(());
        }
        self.map.flush()?;
        // SAFETY: the flag lies in the header, which only this call writes after creation.
        unsafe { self.map.as_mut_ptr().add(COMPLETE_AT).cast::<[u8; 4]>().write_unaligned(1u32.to_le_bytes()) };
        self.map.flush_range(0, HEADER_BYTES)?;
        Ok(())
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        // SAFETY: not used after this point.
        unsafe { ManuallyDrop::drop(&mut self.map) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_path_sits_next_to_the_recording() {
        assert_eq!(pyramid_path(Path::new("/data/a.ap.bin"), "ElectricalSeries/x"), PathBuf::from("/data/cache/a.ap.bin/ElectricalSeries_x.minmax"));
    }
}
