//! Electrophysiology formats through `dspk_synapse::io` (SpikeGLX `.bin` / `.cbin` with their
//! probe, NWB Zarr, mtscomp, plus the generic ones), and each source's probe as a channel layout:
//! groups = shanks, labels = site positions.

use std::path::Path;

use dspk_app_core::plugin::{ChannelGroup, ChannelLayout, SourceOpener};
use dspk_core::{DspResult, RecordingSource};
use dspk_io::SourceEntry;
use dspk_synapse::io::probe::SensorLayout;

pub struct SynapseOpener;

impl SourceOpener for SynapseOpener {
    fn reads(&self, path: &Path) -> bool {
        dspk_synapse::io::detect(path).is_ok()
    }

    fn sources(&self, path: &Path) -> DspResult<Vec<SourceEntry>> {
        dspk_synapse::io::sources(path)
    }

    fn open(&self, path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
        dspk_synapse::io::open_source(path, id)
    }

    fn layout(&self, path: &Path, id: &str) -> Option<ChannelLayout> {
        match dspk_synapse::io::probe::probe_of(path, id) {
            Ok(Some(probe)) => Some(layout_of(&probe)),
            Ok(None) => None,
            Err(e) => {
                tracing::warn!("no probe geometry for {}: {e}", path.display());
                None
            }
        }
    }
}

/// The probe's shanks as groups and its sites as labels (`dspk_synapse::io::probe`:
/// `shank_groups`, `site_labels`), in the app's layout type.
pub fn layout_of(probe: &SensorLayout) -> ChannelLayout {
    let groups = probe.shank_groups().into_iter().map(|g| ChannelGroup { name: g.shank.to_string(), channels: g.channels }).collect();
    ChannelLayout { group_kind: "Shank".into(), groups, labels: probe.site_labels() }
}
