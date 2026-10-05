//! [`Format`] for raw binary files with a JSON sidecar.

use std::path::Path;

use dsp_core::{DspResult, RecordingSource};

use super::{RawParams, RawRecording};
use crate::core::sources::{require_main, single_source, SourceEntry};
use crate::core::Format;

pub struct Raw;

impl Format for Raw {
    fn name(&self) -> &'static str {
        "raw"
    }

    fn detect(&self, path: &Path) -> bool {
        path.is_file() && RawParams::from_sidecar(path).is_some()
    }

    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>> {
        Ok(vec![single_source(&RawRecording::open(path)?)])
    }

    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
        require_main(path, id)?;
        self.open_default(path)
    }

    fn open_default(&self, path: &Path) -> DspResult<Box<dyn RecordingSource>> {
        Ok(Box::new(RawRecording::open(path)?))
    }
}
