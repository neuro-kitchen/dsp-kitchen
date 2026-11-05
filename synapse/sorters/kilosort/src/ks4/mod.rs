//! Kilosort4 (feature `ks4`): the sorter entry point and its provenance. The stages and settings
//! ([`crate::Kilosort4Config`]) are shared with EMUsort and live in the crate root.

use dspk_synapse::provenance::{ArtifactSource, Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};

use crate::runner::{run_plan, Kilosort4Result, RunPlan};
use crate::Kilosort4Config;

/// Paper, code and artifact of Kilosort4.
pub fn kilosort4_provenance() -> Provenance {
    Provenance {
        name: "Kilosort4".into(),
        kind: ProvenanceKind::ReimplementedFromPaperAndCode,
        paper: Some(Paper {
            title: "Spike sorting with Kilosort4".into(),
            authors: vec!["Pachitariu".into(), "Sridhar".into(), "Pennington".into(), "Stringer".into()],
            venue: "Nature Methods".into(),
            year: 2024,
            doi: "10.1038/s41592-024-02232-7".into(),
            license: None,
        }),
        code: UpstreamCode { url: "https://github.com/MouseLand/Kilosort".into(), license: Some("GPL-3.0".into()), version: "v4.1.3".into() },
        artifacts: vec![ArtifactSource {
            name: "wTEMP.npz".into(),
            url: "https://osf.io/download/6807fb5958b763aae139aa60/".into(),
            sha256: "cae1c96f8f4150be0a39627515750b70c4bc3548177cf487ae3c013f1ca6abd8".into(),
            size_bytes: 3432,
            license: None,
        }],
        notes: "Written from the paper and the published defaults, its behaviour matched to upstream's code by reading it (named functions, defaults and stage order are cited in the comments); no code copied, the computation is new (GPU kernels, batching). Implemented: preprocessing, universal templates (learned or predefined), universal-template detection, graph-based clustering, learned templates, learned-template matching and the final clustering. Also the refractory (cross-correlogram) split criterion, global merges, duplicate-spike removal and good / mua labels. Not yet: drift correction. The predefined wTEMP.npz is only used when templates_from_data is false.".into(),
    }
}

/// Kilosort4 over a whole recording ([`run_plan`] with [`RunPlan::kilosort4`]).
#[derive(Debug, Clone, Default)]
pub struct Kilosort4 {
    pub config: Kilosort4Config,
}

impl Kilosort4 {
    pub fn new(config: Kilosort4Config) -> Self {
        Self { config }
    }

    /// Runs Kilosort4 over `source` on `client`'s device runtime.
    pub fn run(
        &self,
        client: &cubecl::prelude::Client,
        source: &dyn dspk_core::RecordingSource,
        probe: &dspk_synapse::io::probe::SensorLayout,
    ) -> dspk_core::DspResult<Kilosort4Result> {
        self.run_with_progress(client, source, probe, &dspk_core::NoProgress)
    }

    /// [`Self::run`], reporting each stage to `progress` ([`crate::runner::STAGE_FIT`] …).
    pub fn run_with_progress(
        &self,
        client: &cubecl::prelude::Client,
        source: &dyn dspk_core::RecordingSource,
        probe: &dspk_synapse::io::probe::SensorLayout,
        progress: &dyn dspk_core::ProgressSink,
    ) -> dspk_core::DspResult<Kilosort4Result> {
        run_plan(client, source, probe, &RunPlan::kilosort4(&self.config), progress)
    }
}

impl Attributed for Kilosort4 {
    fn provenance(&self) -> Provenance {
        kilosort4_provenance()
    }
}
