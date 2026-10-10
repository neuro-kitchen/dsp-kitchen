//! Kilosort4 (Pachitariu, Sridhar, Pennington & Stringer, Nature Methods 2024), written from the
//! paper and the published defaults — the upstream GPL-3.0 code is not ported.
//!
//! Implemented stages: universal templates (`wPCA` / `wTEMP`) learned from the recording or loaded
//! from the predefined `wTEMP.npz` ([`templates`]), universal-template spike detection with `wPCA`
//! features on the device ([`detect`]), and the first graph-based clustering of those spikes into
//! units ([`clustering`]), learned templates ([`learned`]), learned-template matching
//! ([`matching`]), the clustering of the matched spikes with the refractory criterion, global
//! merges ([`merges`]), duplicate-spike removal and good / mua labels, driven over a whole
//! recording by [`runner`] (preprocessing fit: CAR, high-pass, local whitening). Not yet: drift
//! correction. See the book's *Sorters* pages.

pub mod clustering;
pub mod detect;
pub mod learned;
pub mod matching;
pub mod merges;
mod kernels;
pub mod runner;
pub mod templates;

pub use clustering::{cluster_spikes, ClusteringOptions, SpikeClusters};
pub use matching::{TemplateMatcher, MAX_PEELS};
pub use merges::{global_merges, GlobalMergeOptions, MergedClusters};
pub use learned::{learned_templates, LearnedTemplates, TemplateMergeOptions};
pub use detect::{detect_universal, CentreOptions, TemplateCentres, UniversalDetector, UniversalSpike};
pub use runner::{
    fit_kilosort4_preprocessing, fit_preprocessing, run_plan, ChannelDelays, FitSettings, FittedPreprocessing, Kilosort4Result,
    RunPlan,
};
pub use templates::{
    extract_clips, learn_universal_templates, learn_universal_templates_with_progress, ClipOptions, ClipScaling, LearnOptions, UniversalTemplates,
};

use dsp_synapse::NeuralBand;

use crate::provenance::{ArtifactSource, Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};

/// Shortest waveform `nt_ms` can give (samples).
const MIN_NT: usize = 3;

/// Detection thresholds (whitened σ) for the default 300–6000 Hz band. Upstream Kilosort4 uses 9
/// and 8 with a high-pass only; the band's upper edge lowers the whitened noise, so the same
/// thresholds detect ~36% more spikes. 10 and 9 give the same agreement with Kilosort4's saved
/// results as 9 and 8 with the high-pass (test recording, 2026-10-09; 11 and 10 miss spikes).
pub const TH_UNIVERSAL: f32 = 10.0;
pub const TH_LEARNED: f32 = 9.0;
/// Upstream Kilosort4's thresholds (high-pass only, `bandpass_high_hz = None`).
pub const UPSTREAM_TH_UNIVERSAL: f32 = 9.0;
pub const UPSTREAM_TH_LEARNED: f32 = 8.0;

/// Line-noise frequency of the notch (Hz): 60 (50 outside the Americas).
pub const LINE_NOISE_HZ: f64 = 60.0;

/// Default quality factor of the line-noise notch (−3 dB bandwidth `f / 30`: 2 Hz at 60 Hz).
pub const NOTCH_Q: f64 = 30.0;

/// Catalog id of Kilosort4's predefined `wTEMP.npz`.
pub const KILOSORT4_WTEMP_MODEL_ID: &str = "kilosort4/wtemp-v1";

/// Kilosort4 settings this crate uses, with upstream defaults (`kilosort/parameters.py`).
#[derive(Debug, Clone, PartialEq)]
pub struct Kilosort4Config {
    /// Samples per waveform (odd). Used unless `nt_ms` is set.
    pub nt: usize,
    /// Waveform length in ms: when set, `nt` is `round(nt_ms · fs / 1000)` made odd for each
    /// recording (and `nt0min`, unless set, follows it), so one setting fits every sampling rate.
    pub nt_ms: Option<f64>,
    /// Sample a waveform's peak is aligned to; `None`: `int(20 · nt / 61)`.
    pub nt0min: Option<usize>,
    /// Universal-template detection threshold (whitened σ).
    pub th_universal: f32,
    /// Learned-template matching threshold (whitened σ).
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
    /// Butterworth band-pass `bandpass_low_hz .. bandpass_high_hz` (order 3 per edge,
    /// forward-backward); off: no Butterworth (data already filtered).
    pub do_bandpass: bool,
    /// Band edges (Hz). Default: the action-potential band, [`NeuralBand::Ap`] (300–6000 Hz). The
    /// upper edge must be below Nyquist; `None`: no upper edge, a high-pass at `bandpass_low_hz`
    /// (upstream Kilosort4's filter).
    pub bandpass_low_hz: f64,
    pub bandpass_high_hz: Option<f64>,
    /// Line-noise notch after the band (off by default for Kilosort4).
    pub do_notch: bool,
    /// Notch frequency (Hz; 60, or 50 outside the Americas) and quality factor (−3 dB bandwidth
    /// `notch_hz / notch_q`).
    pub notch_hz: f64,
    pub notch_q: f64,
    /// Channels in each local whitening neighbourhood.
    pub whitening_range: usize,
    /// Samples per batch.
    pub batch_size: usize,
    /// Clustering of the detected spikes into units ([`clustering`]).
    pub clustering: ClusteringOptions,
    /// Alignment and merging of the units' templates into the learned templates ([`learned`]).
    pub template_merge: TemplateMergeOptions,
    /// Matching pursuit rounds per window of learned-template matching ([`matching`]).
    pub max_peels: usize,
    /// Global merges of the final units ([`merges`]).
    pub global_merge: GlobalMergeOptions,
    /// Spikes of one unit within this many ms of its previous spike are duplicates and removed
    /// (`duplicate_spike_ms`; 0 keeps them). EMUsort's `duplicate_spike_bins = 7` is the same
    /// setting in samples (0.23 ms at 30 kHz).
    pub duplicate_spike_ms: f64,
    /// Same input on the same device gives the same spikes: the autotuned choices that change the
    /// numbers (IIR time blocks, matrix-product routine) are pinned for the run
    /// ([`dsp_core::compute::pin_tuned_choices`]). Off, the tuner may pick differently between runs
    /// (last-bit differences that can move a spike across a threshold).
    pub reproducible: bool,
}

impl Default for Kilosort4Config {
    fn default() -> Self {
        Self {
            nt: 61,
            nt_ms: None,
            nt0min: None,
            th_universal: TH_UNIVERSAL,
            th_learned: TH_LEARNED,
            th_single_ch: vec![6.0],
            templates_from_data: true,
            n_templates: 6,
            n_pcs: 6,
            nskip: 25,
            centres: CentreOptions::default(),
            do_car: true,
            do_bandpass: true,
            bandpass_low_hz: NeuralBand::Ap.low_hz(),
            bandpass_high_hz: Some(NeuralBand::Ap.high_hz()),
            do_notch: false,
            notch_hz: LINE_NOISE_HZ,
            notch_q: NOTCH_Q,
            whitening_range: 32,
            batch_size: 60_000,
            clustering: ClusteringOptions::default(),
            template_merge: TemplateMergeOptions::default(),
            max_peels: MAX_PEELS,
            global_merge: GlobalMergeOptions::default(),
            duplicate_spike_ms: 0.25,
            reproducible: true,
        }
    }
}

impl Kilosort4Config {
    /// These settings for a recording at `sample_rate_hz`: with `nt_ms`, `nt` becomes
    /// `round(nt_ms · fs / 1000)`, made odd (at least 3), and `nt_ms` is cleared.
    pub fn resolved(&self, sample_rate_hz: f64) -> Self {
        let mut c = self.clone();
        if let Some(ms) = c.nt_ms.take() {
            let n = ((ms * sample_rate_hz / 1000.0).round() as usize).max(MIN_NT);
            c.nt = n | 1;
        }
        c
    }

    /// `nt0min`, defaulting to `int(20 · nt / 61)`.
    pub fn nt0min(&self) -> usize {
        self.nt0min.unwrap_or(20 * self.nt / 61)
    }

    pub fn clip_options(&self) -> ClipOptions {
        ClipOptions { nt: self.nt, nt0min: self.nt0min(), thresholds: self.th_single_ch.clone() }
    }

    pub fn learn_options(&self) -> LearnOptions {
        LearnOptions {
            n_pcs: self.n_pcs,
            n_templates: self.n_templates,
            clip_scaling: ClipScaling::PerClip,
            outlier_min_cluster_size: None,
            seed: 0,
        }
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
        notes: "Written from the paper and the published defaults; GPL code not ported. Implemented: preprocessing, universal templates (learned or predefined), universal-template detection, graph-based clustering, learned templates, learned-template matching and the final clustering. Also the refractory (cross-correlogram) split criterion, global merges, duplicate-spike removal and good / mua labels. Not yet: drift correction. The predefined wTEMP.npz is only used when templates_from_data is false.".into(),
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
        source: &dyn dsp_core::RecordingSource,
        probe: &dsp_io::neuro::probe::SensorLayout,
    ) -> dsp_core::DspResult<Kilosort4Result> {
        self.run_with_progress(client, source, probe, &dsp_core::NoProgress)
    }

    /// [`Self::run`], reporting each stage to `progress` ([`runner::STAGE_FIT`] …).
    pub fn run_with_progress(
        &self,
        client: &cubecl::prelude::Client,
        source: &dyn dsp_core::RecordingSource,
        probe: &dsp_io::neuro::probe::SensorLayout,
        progress: &dyn dsp_core::ProgressSink,
    ) -> dsp_core::DspResult<Kilosort4Result> {
        run_plan(client, source, probe, &RunPlan::kilosort4(&self.config), progress)
    }
}

impl Attributed for Kilosort4 {
    fn provenance(&self) -> Provenance {
        kilosort4_provenance()
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn nt_ms_gives_an_odd_window_per_sampling_rate() {
        let c = Kilosort4Config { nt_ms: Some(5.0), ..Default::default() };
        // 5 ms: 150 → 151 at 30 kHz, 122.07 → 122 → 123 at 24.4 kHz
        let at30 = c.resolved(30_000.0);
        assert_eq!((at30.nt, at30.nt_ms), (151, None));
        assert_eq!(at30.nt0min(), 20 * 151 / 61);
        assert_eq!(c.resolved(24_414.0625).nt, 123);
        // Without nt_ms, nt stands
        assert_eq!(Kilosort4Config::default().resolved(24_414.0625).nt, 61);
    }
}
