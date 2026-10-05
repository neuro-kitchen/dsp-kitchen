//! Probe geometry read from recording files, alongside (not inside) the recording.

use std::path::Path;

use dsp_core::DspResult;

use super::SensorLayout;
use crate::core::Format;

/// A [`Format`] whose files can describe the probe a source was recorded with.
pub trait ProbeSource: Format {
    /// Geometry of source `id` of `path` (an id from [`Format::sources`]); `None` when the file
    /// has none for it.
    fn probe(&self, path: &Path, id: &str) -> DspResult<Option<SensorLayout>>;
}

/// Formats that implement [`ProbeSource`], in detection order.
static PROBE_SOURCES: &[&dyn ProbeSource] = &[&crate::neuro::spikeglx::SpikeGlx];

/// Probe geometry of source `id` of `path`; `None` when its format carries none. Sites are numbered
/// by the source's channels; apply [`SensorLayout::select_channels`] for a channel subset.
pub fn probe_of(path: &Path, id: &str) -> DspResult<Option<SensorLayout>> {
    match PROBE_SOURCES.iter().find(|f| f.detect(path)) {
        Some(f) => f.probe(path, id),
        None => Ok(None),
    }
}
