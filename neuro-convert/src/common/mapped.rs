//! Read-only memory-mapped files.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{Error, Result};

/// A whole file mapped read-only.
pub struct MappedFile {
    map: Mmap,
}

impl MappedFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        // SAFETY: read-only map; like every mmap reader we assume the recording is not
        // truncated by another process while it is open.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(path, e))?;
        Ok(Self { map })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.map
    }

    pub fn len(&self) -> u64 {
        self.map.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
