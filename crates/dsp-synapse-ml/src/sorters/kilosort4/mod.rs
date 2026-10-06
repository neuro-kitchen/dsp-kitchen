//! Kilosort4 (Pachitariu, Sridhar, Pennington & Stringer, Nature Methods 2024), written from the
//! paper and the published defaults — the upstream GPL-3.0 code is not ported.
//!
//! Implemented stages: universal templates (`wPCA` / `wTEMP`) learned from the recording or loaded
//! from the predefined `wTEMP.npz` ([`templates`]), and universal-template spike detection with
//! `wPCA` features on the device ([`detect`]), driven over a whole recording by [`runner`]
//! (preprocessing fit: CAR, high-pass, local whitening). Not yet: drift correction, graph-based
//! clustering, learned-template deconvolution, merging. See the book's *Sorters* pages.

pub mod detect;
mod kernels;
pub mod runner;
pub mod templates;

pub use detect::{detect_universal, CentreOptions, TemplateCentres, UniversalDetector, UniversalSpike};
pub use runner::{
    fit_kilosort4_preprocessing, fit_preprocessing, run_plan, ChannelDelays, FitSettings, FittedPreprocessing, Kilosort4Result,
    RunPlan,
};
pub use templates::{extract_clips, learn_universal_templates, ClipOptions, LearnOptions, UniversalTemplates};

use crate::provenance::{ArtifactSource, Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};

/// Catalog id of Kilosort4's predefined `wTEMP.npz`.
pub const KILOSORT4_WTEMP_MODEL_ID: &str = "kilosort4/wtemp-v1";

/// Kilosort4 settings this crate uses, with upstream defaults (`kilosort/parameters.py`).
#[derive(Debug, Clone, PartialEq)]
pub struct Kilosort4Config {
    /// Samples per waveform (odd).
    pub nt: usize,
    /// Sample a waveform's peak is aligned to; `None`: `int(20 · nt / 61)`.
    pub nt0min: Option<usize>,
    /// Universal-template detection threshold (whitened σ).
    pub th_universal: f32,
    /// Learned-template detection threshold (whitened σ; used by deconvolution, not yet here).
    pub th_learned: f32,
    /// Single-channel thresholds for the clips the universal templates are learned from.
    pub th_single_ch: Vec<f32>,
    /// Learn `wPCA` / `wTEMP` from the recording (else load the predefined `wTEMP.npz`).
    pub templates_from_data: bool,
    pub n_templates: usize,
    pub n_pcs: usize,
    /// Every `nskip`-th batch is used for template learning.
    pub nskip: usize,
    pub centres: CentreOptions,
    /// Common average reference before filtering.
    pub do_car: bool,
    pub highpass_cutoff_hz: f64,
    /// Channels in each local whitening neighbourhood.
    pub whitening_range: usize,
    /// Samples per batch.
    pub batch_size: usize,
}

impl Default for Kilosort4Config {
    fn default() -> Self {
        Self {
            nt: 61,
            nt0min: None,
            th_universal: 9.0,
            th_learned: 8.0,
            th_single_ch: vec![6.0],
            templates_from_data: true,
            n_templates: 6,
            n_pcs: 6,
            nskip: 25,
            centres: CentreOptions::default(),
            do_car: true,
            highpass_cutoff_hz: 300.0,
            whitening_range: 32,
            batch_size: 60_000,
        }
    }
}

impl Kilosort4Config {
    /// `nt0min`, defaulting to `int(20 · nt / 61)`.
    pub fn nt0min(&self) -> usize {
        self.nt0min.unwrap_or(20 * self.nt / 61)
    }

    pub fn clip_options(&self) -> ClipOptions {
        ClipOptions { nt: self.nt, nt0min: self.nt0min(), thresholds: self.th_single_ch.clone() }
    }

    pub fn learn_options(&self) -> LearnOptions {
        LearnOptions { n_pcs: self.n_pcs, n_templates: self.n_templates, outlier_min_cluster_size: None, seed: 0 }
    }
}

/// Paper, code and artifact of Kilosort4.
pub fn kilosort4_provenance() -> Provenance {
    Provenance {
        name: "Kilosort4".into(),
        kind: ProvenanceKind::ReimplementedFromPaper,
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
        notes: "Written from the paper and the published defaults; GPL code not ported. Implemented: universal templates (learned or predefined) and universal-template detection. The predefined wTEMP.npz is only used when templates_from_data is false.".into(),
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
    pub fn run<R: cubecl::prelude::Runtime>(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        source: &dyn dsp_core::RecordingSource,
        probe: &dsp_io::neuro::probe::SensorLayout,
    ) -> dsp_core::DspResult<Kilosort4Result> {
        run_plan(client, source, probe, &RunPlan::kilosort4(&self.config))
    }
}

impl Attributed for Kilosort4 {
    fn provenance(&self) -> Provenance {
        kilosort4_provenance()
    }
}
