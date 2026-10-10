//! MountainSort 5 (Chung et al., Neuron 2017; `flatironinstitute/mountainsort5`, Apache-2.0),
//! ported from its source: locally exclusive detection ([`detect`]), masked snippets
//! ([`snippets`]), PCA and the isosplit6 subdivision method ([`subdivision`],
//! [`dsp_synapse::sorting::isosplit6`]), median templates and alignment ([`templates`],
//! [`scheme1`]), then per-channel classifiers over the whole recording ([`scheme2`]), driven by
//! [`runner`]. Stages, settings and our choices: the book's MountainSort 5 pages.

pub mod detect;
pub mod kernels;
pub mod runner;
pub mod scheme1;
pub mod scheme2;
pub mod snippets;
pub mod subdivision;
pub mod templates;

pub use detect::{DetectOptions, Detections, Detector};
pub use runner::{run, Mountainsort5Result, MOUNTAINSORT5_SORTER};
pub use scheme1::{cluster_events, cluster_events_with_progress, ClusteringParams, Events, Scheme1Output};
pub use scheme2::{ClassifierParams, Classifiers};
pub use snippets::{MaskedSnippets, SnippetRows};
pub use subdivision::{isosplit6_subdivision, isosplit6_subdivision_with_progress, SubdivisionOptions};

use dsp_base::linalg::TopComponentsOptions;
use dsp_synapse::sorting::IsosplitOptions;

use crate::provenance::{Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};

/// One pass (scheme 1) or train on a stretch and classify everything (scheme 2, the default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scheme {
    One,
    #[default]
    Two,
}

/// Where scheme 2's training stretch comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingSampling {
    /// The first `training_duration_sec`.
    Initial,
    /// 10 s chunks spread evenly over the recording.
    #[default]
    Uniform,
}

/// MountainSort 5 settings: the package's defaults, and SpikeInterface's wrapper's where the
/// package leaves them to the caller (book: MountainSort 5 parameters).
#[derive(Debug, Clone, PartialEq)]
pub struct Mountainsort5Config {
    pub scheme: Scheme,
    /// Common average reference first (not in the wrapper: off).
    pub do_car: bool,
    /// Band-pass (a high-pass without an upper edge): SpikeInterface's `bandpass_filter`,
    /// Butterworth order 5, forward-backward.
    pub do_bandpass: bool,
    pub bandpass_low_hz: f64,
    pub bandpass_high_hz: Option<f64>,
    pub do_notch: bool,
    pub notch_hz: f64,
    pub notch_q: f64,
    /// Global ZCA whitening (SpikeInterface's `whiten`).
    pub do_whiten: bool,
    /// The whitening is fitted on this many chunks of `whitening_chunk_ms`, evenly spaced.
    pub whitening_chunks: usize,
    pub whitening_chunk_ms: f64,
    /// SpikeInterface's `eps` for data in µV (median square ≥ 1); values below
    /// `dsp_base::spatial::MIN_WHITENING_EPSILON` act as that floor.
    pub whitening_epsilon: f32,
    pub detect_threshold: f32,
    /// −1 negative peaks, +1 positive, 0 both.
    pub detect_sign: i8,
    pub detect_time_radius_ms: f64,
    /// Scheme 1's detection neighbourhood (µm; `None`: all channels).
    pub scheme1_detect_channel_radius_um: Option<f32>,
    /// Scheme 2's phase 1 (training) detection.
    pub phase1_detect_channel_radius_um: Option<f32>,
    pub phase1_detect_threshold: f32,
    pub phase1_detect_time_radius_ms: f64,
    /// Scheme 2's phase 2 (classification) detection neighbourhood.
    pub detect_channel_radius_um: Option<f32>,
    pub snippet_t1: usize,
    pub snippet_t2: usize,
    pub snippet_mask_radius_um: Option<f32>,
    pub npca_per_channel: usize,
    pub npca_per_subdivision: usize,
    pub skip_alignment: bool,
    pub isocut_threshold: f64,
    pub min_cluster_size: usize,
    pub k_init: usize,
    pub max_iterations_per_pass: usize,
    /// `None`: the whole recording.
    pub training_duration_sec: Option<f64>,
    pub training_sampling: TrainingSampling,
    pub max_num_snippets_per_training_batch: usize,
    /// `None`: `max(12, 3 · mask channels)`.
    pub classifier_npca: Option<usize>,
    /// Samples per window; `None`: `⌈10⁸ / channels⌉` (upstream's classification chunk).
    pub classification_chunk_sec: Option<f64>,
    /// Above this many features PCA is randomized (scikit-learn's `covariance_eigh` cap).
    pub pca_exact_cap: usize,
    pub seed: u64,
}

impl Default for Mountainsort5Config {
    fn default() -> Self {
        Self {
            scheme: Scheme::Two,
            do_car: false,
            do_bandpass: true,
            bandpass_low_hz: 300.0,
            bandpass_high_hz: Some(6000.0),
            do_notch: false,
            notch_hz: 60.0,
            notch_q: 30.0,
            do_whiten: true,
            whitening_chunks: 20,
            whitening_chunk_ms: 500.0,
            whitening_epsilon: 1e-16,
            detect_threshold: 5.5,
            detect_sign: -1,
            detect_time_radius_ms: 0.5,
            scheme1_detect_channel_radius_um: Some(150.0),
            phase1_detect_channel_radius_um: Some(200.0),
            phase1_detect_threshold: 5.5,
            phase1_detect_time_radius_ms: 1.5,
            detect_channel_radius_um: Some(50.0),
            snippet_t1: 20,
            snippet_t2: 20,
            snippet_mask_radius_um: Some(250.0),
            npca_per_channel: 3,
            npca_per_subdivision: 10,
            skip_alignment: false,
            isocut_threshold: 2.0,
            min_cluster_size: 10,
            k_init: 200,
            max_iterations_per_pass: 500,
            training_duration_sec: Some(300.0),
            training_sampling: TrainingSampling::Uniform,
            max_num_snippets_per_training_batch: 200,
            classifier_npca: None,
            classification_chunk_sec: None,
            pca_exact_cap: 8000,
            seed: 0,
        }
    }
}

impl Mountainsort5Config {
    pub fn pca(&self) -> TopComponentsOptions {
        TopComponentsOptions { exact_cap: self.pca_exact_cap, seed: self.seed, ..Default::default() }
    }

    pub fn clustering(&self) -> ClusteringParams {
        ClusteringParams {
            npca_per_channel: self.npca_per_channel,
            subdivision: SubdivisionOptions {
                npca_per_subdivision: self.npca_per_subdivision,
                isosplit: IsosplitOptions {
                    isocut_threshold: self.isocut_threshold,
                    min_cluster_size: self.min_cluster_size,
                    k_init: self.k_init,
                    max_iterations_per_pass: self.max_iterations_per_pass,
                    variant: dsp_synapse::sorting::IsosplitVariant::Isosplit6,
                },
                pca: self.pca(),
            },
            skip_alignment: self.skip_alignment,
            detect_sign: self.detect_sign,
            pca: self.pca(),
        }
    }

    pub fn classifier(&self) -> ClassifierParams {
        ClassifierParams {
            detect_threshold: self.detect_threshold,
            detect_sign: self.detect_sign,
            max_num_snippets_per_training_batch: self.max_num_snippets_per_training_batch,
            classifier_npca: self.classifier_npca,
            // Randomized (module docs of `scheme2`): only the subspace matters to the distances
            pca: TopComponentsOptions { exact_cap: 0, ..self.pca() },
        }
    }
}

/// Paper and code of MountainSort 5.
pub fn mountainsort5_provenance() -> Provenance {
    Provenance {
        name: "MountainSort 5".into(),
        kind: ProvenanceKind::PortedFromCode,
        paper: Some(Paper {
            title: "A fully automated approach to spike sorting".into(),
            authors: vec!["Chung".into(), "Magland".into(), "Barnett".into(), "Tolosa".into(), "Tooker".into(), "Lee".into(), "Shah".into(), "Felix".into(), "Frank".into(), "Greengard".into()],
            venue: "Neuron".into(),
            year: 2017,
            doi: "10.1016/j.neuron.2017.08.030".into(),
            license: None,
        }),
        code: UpstreamCode {
            url: "https://github.com/flatironinstitute/mountainsort5".into(),
            license: Some("Apache-2.0".into()),
            version: "v0.5.9 (isosplit6: magland/isosplit6)".into(),
        },
        artifacts: Vec::new(),
        notes: "Ported stage by stage from the Apache-2.0 source: detection, masked snippets, PCA, isosplit6 subdivision, median templates and alignment (scheme 1), per-channel classifiers (scheme 2). Preprocessing as SpikeInterface's wrapper (band-pass 300-6000 Hz, global whitening). Our choices are listed in the book.".into(),
    }
}

/// MountainSort 5 over a whole recording ([`run`]).
#[derive(Debug, Clone, Default)]
pub struct Mountainsort5 {
    pub config: Mountainsort5Config,
}

impl Mountainsort5 {
    pub fn new(config: Mountainsort5Config) -> Self {
        Self { config }
    }

    /// Runs MountainSort 5 over `source` on `client`'s device runtime.
    pub fn run(
        &self,
        client: &cubecl::prelude::Client,
        source: &dyn dsp_core::RecordingSource,
        probe: &dsp_io::neuro::probe::SensorLayout,
    ) -> dsp_core::DspResult<Mountainsort5Result> {
        run(client, source, probe, &self.config, &dsp_core::NoProgress)
    }

    pub fn run_with_progress(
        &self,
        client: &cubecl::prelude::Client,
        source: &dyn dsp_core::RecordingSource,
        probe: &dsp_io::neuro::probe::SensorLayout,
        progress: &dyn dsp_core::ProgressSink,
    ) -> dsp_core::DspResult<Mountainsort5Result> {
        run(client, source, probe, &self.config, progress)
    }
}

impl Attributed for Mountainsort5 {
    fn provenance(&self) -> Provenance {
        mountainsort5_provenance()
    }
}
