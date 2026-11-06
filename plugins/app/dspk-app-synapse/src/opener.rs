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

/// Shanks as groups (ascending), sites as labels (`S1 · 16,40 µm` on several shanks).
pub fn layout_of(probe: &SensorLayout) -> ChannelLayout {
    let channels = probe.contacts.iter().map(|s| s.channel_id + 1).max().unwrap_or(0);
    let mut shanks: Vec<usize> = probe.contacts.iter().map(|s| s.shank_id).collect();
    shanks.sort_unstable();
    shanks.dedup();
    let multi = shanks.len() > 1;
    let mut labels = vec![None; channels];
    for c in 0..channels {
        if let Ok(site) = probe.get_site(c) {
            let pos = format!("{:.0},{:.0} µm", site.position.x_um, site.position.y_um);
            labels[c] = Some(if multi { format!("S{} · {pos}", site.shank_id) } else { pos });
        }
    }
    let groups = shanks
        .iter()
        .map(|&sh| {
            // Along the shank (by depth), as the probe is laid out
            let mut sites: Vec<_> = probe.contacts.iter().filter(|s| s.shank_id == sh).collect();
            sites.sort_by(|a, b| a.position.y_um.total_cmp(&b.position.y_um).then(a.position.x_um.total_cmp(&b.position.x_um)));
            ChannelGroup { name: sh.to_string(), channels: sites.iter().map(|s| s.channel_id).collect() }
        })
        .collect();
    ChannelLayout { group_kind: "Shank".into(), groups, labels }
}
