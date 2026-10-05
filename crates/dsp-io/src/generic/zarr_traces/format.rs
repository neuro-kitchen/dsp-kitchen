//! [`Format`] for Zarr v3 stores with a `/traces` array.

use std::path::Path;

use dsp_core::{DspResult, RecordingSource};

use super::ZarrRecording;
use crate::core::sources::{require_main, single_source, SourceEntry};
use crate::core::Format;

pub struct ZarrTraces;

impl Format for ZarrTraces {
    fn name(&self) -> &'static str {
        "zarr-traces"
    }

    fn detect(&self, path: &Path) -> bool {
        (path.is_dir() || path.extension().is_some_and(|e| e == "zarr"))
            && path.join("zarr.json").exists()
            && path.join("traces").join("zarr.json").exists()
    }

    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>> {
        Ok(vec![single_source(&ZarrRecording::open(path)?)])
    }

    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
        require_main(path, id)?;
        self.open_default(path)
    }

    fn open_default(&self, path: &Path) -> DspResult<Box<dyn RecordingSource>> {
        Ok(Box::new(ZarrRecording::open(path)?))
    }
}
