//! [`Format`] for mtscomp-compressed recordings (`.cbin` + `.ch`) without a SpikeGLX `.meta`.

use std::path::Path;

use dsp_core::{DspResult, RecordingSource};

use super::MtscompRecording;
use crate::core::sources::{require_main, single_source, SourceEntry};
use crate::core::Format;

pub struct Mtscomp;

impl Format for Mtscomp {
    fn name(&self) -> &'static str {
        "mtscomp"
    }

    fn detect(&self, path: &Path) -> bool {
        path.is_file() && path.extension().is_some_and(|e| e == "cbin")
    }

    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>> {
        Ok(vec![single_source(&MtscompRecording::open(path)?)])
    }

    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
        require_main(path, id)?;
        self.open_default(path)
    }

    fn open_default(&self, path: &Path) -> DspResult<Box<dyn RecordingSource>> {
        Ok(Box::new(MtscompRecording::open(path)?))
    }
}
