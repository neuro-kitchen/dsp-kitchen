//! [`Format`] for WAV files (detected by their `.wav` name, read with their header).

use std::path::Path;

use dspk_core::{DspResult, RecordingSource};

use super::{is_wav_name, open_wav};
use crate::core::sources::{require_main, single_source, SourceEntry};
use crate::core::Format;

pub struct Wav;

impl Format for Wav {
    fn name(&self) -> &'static str {
        "wav"
    }

    fn detect(&self, path: &Path) -> bool {
        path.is_file() && is_wav_name(&path.to_string_lossy())
    }

    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>> {
        Ok(vec![single_source(&open_wav(path)?)])
    }

    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
        require_main(path, id)?;
        self.open_default(path)
    }

    fn open_default(&self, path: &Path) -> DspResult<Box<dyn RecordingSource>> {
        Ok(Box::new(open_wav(path)?))
    }
}
