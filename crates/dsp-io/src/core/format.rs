//! The [`Format`] trait every file format implements.

use std::path::Path;

use dsp_core::{DspError, DspResult, RecordingSource};

use super::sources::{default_source, SourceEntry};

/// One file format. Implemented by a unit struct in the format's folder and listed in
/// [`crate::registry::formats`].
pub trait Format: Send + Sync {
    /// Stable name (`"raw"`, `"zarr-traces"`, `"nwb"`, `"spikeglx"`, `"mtscomp"`).
    fn name(&self) -> &'static str;

    /// Whether `path` is in this format, from its path and metadata only (never sample data).
    fn detect(&self, path: &Path) -> bool;

    /// The signals `path` offers, from metadata only.
    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>>;

    /// Opens source `id` (an id from [`Self::sources`]).
    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>>;

    /// Opens the source to show first ([`default_source`] of [`Self::sources`]). Formats with a
    /// cheaper way override it.
    fn open_default(&self, path: &Path) -> DspResult<Box<dyn RecordingSource>> {
        let list = self.sources(path)?;
        let entry = default_source(&list)
            .ok_or_else(|| DspError::UnsupportedFormat(format!("{} has no sources", path.display())))?;
        self.open(path, &entry.id)
    }
}
