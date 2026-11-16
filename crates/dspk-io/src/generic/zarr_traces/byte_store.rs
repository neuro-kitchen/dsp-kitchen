//! A read-only Zarr store over named byte sources: a Zarr folder whose files are not on a disk
//! this process can open by path (the files a browser page picked, read in a worker), or are held
//! in memory. Only the byte ranges zarrs asks for are read, so a chunk (or a sharded chunk's
//! inner chunk) is read without the rest of its file.

use std::collections::HashMap;
use std::sync::Arc;

use zarrs::storage::byte_range::{ByteRangeIterator, InvalidByteRangeError};
use zarrs::storage::{Bytes, MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey};

use crate::generic::raw::ByteSource;

/// A Zarr store whose keys (`zarr.json`, `traces/zarr.json`, `traces/c/0/0`, …) are byte sources.
pub struct ByteSourceStore {
    files: HashMap<String, Arc<dyn ByteSource>>,
}

impl ByteSourceStore {
    /// The store holding `files`, by their key: their path inside the Zarr folder, with `/`.
    pub fn new(files: impl IntoIterator<Item = (String, Arc<dyn ByteSource>)>) -> Self {
        Self { files: files.into_iter().collect() }
    }
}

impl ReadableStorageTraits for ByteSourceStore {
    fn get_partial_many<'a>(&'a self, key: &StoreKey, byte_ranges: ByteRangeIterator<'a>) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let Some(source) = self.files.get(key.as_str()).cloned() else { return Ok(None) };
        let size = source.len();
        Ok(Some(Box::new(byte_ranges.map(move |range| {
            let (start, end) = (range.start(size), range.end(size));
            if end > size || start > end {
                return Err(InvalidByteRangeError::new(range, size).into());
            }
            let mut bytes = vec![0u8; (end - start) as usize];
            source.read_at(start, &mut bytes).map_err(|e| StorageError::from(Arc::new(e)))?;
            Ok(Bytes::from(bytes))
        }))))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        Ok(self.files.get(key.as_str()).map(|source| source.len()))
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

#[cfg(all(test, feature = "zarr"))]
mod tests {
    use super::*;
    use crate::{write_zarr, ZarrCompression, ZarrOptions, ZarrRecording};
    use dspk_core::{MemoryRecording, RecordingSource};

    struct InMemory(Vec<u8>);
    impl ByteSource for InMemory {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
        fn read_at(&self, offset: u64, out: &mut [u8]) -> std::io::Result<()> {
            let at = offset as usize;
            out.copy_from_slice(self.0.get(at..at + out.len()).ok_or_else(|| std::io::Error::from(std::io::ErrorKind::UnexpectedEof))?);
            Ok(())
        }
    }

    /// Every file under `dir`, keyed by its path inside it (with `/`).
    fn files_of(dir: &std::path::Path) -> Vec<(String, Arc<dyn ByteSource>)> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for entry in std::fs::read_dir(&d).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let key = path.strip_prefix(dir).unwrap().components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/");
                    out.push((key, Arc::new(InMemory(std::fs::read(&path).unwrap())) as Arc<dyn ByteSource>));
                }
            }
        }
        out
    }

    /// A Zarr recording read through byte sources (as a browser's engine reads the files picked)
    /// reads as the same folder opened on disk, for each compression dspk writes.
    #[test]
    fn a_zarr_recording_reads_through_byte_sources_as_from_disk() {
        let data: Vec<f32> = (0..3 * 2500).map(|i| ((i % 997) as f32 * 0.31).sin() * 50.0).collect();
        let source = MemoryRecording::new("rec", data, 3, 30_000.0).unwrap();
        for (i, compression) in [ZarrCompression::None, ZarrCompression::Zstd { level: 3 }, ZarrCompression::BloscLz4 { level: 5 }].into_iter().enumerate() {
            let dir = std::env::temp_dir().join(format!("dspk_io_byte_store_{}_{i}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let path = dir.join("rec.zarr");
            write_zarr(&source, &path, &ZarrOptions { chunk_samples: 700, compression, ..Default::default() }, |_, _| {}).unwrap();
            let on_disk = ZarrRecording::open(&path).unwrap();
            let through = ZarrRecording::from_store("rec.zarr", Arc::new(ByteSourceStore::new(files_of(&path)))).unwrap();
            assert_eq!(through.info().samples, on_disk.info().samples);
            assert_eq!(through.info().sample_rate_hz(), on_disk.info().sample_rate_hz());
            let (mut a, mut b) = (vec![0.0f32; 2 * 900], vec![0.0f32; 2 * 900]);
            through.read(&[2, 0], 600..1500, &mut a).unwrap();
            on_disk.read(&[2, 0], 600..1500, &mut b).unwrap();
            assert_eq!(a, b, "{compression:?}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
