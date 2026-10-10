//! Tridesclous 2 (Garcia & Pouzat's Tridesclous, rewritten in SpikeInterface: `sorters/internal/
//! tridesclous2.py`, MIT, sorter version 2026.01), ported from its source: Bessel band-pass, common
//! median reference, local whitening, `locally_exclusive` detection, local SVD features, iterative
//! splits with SpikeInterface's isosplit, template cleaning and merging, mean templates, the
//! `tdc-peeler` matching and the final merges ([`super::components`]), driven by [`runner`].
//!
//! Not yet: motion correction (off by default upstream too). Stages and settings: the book's
//! Tridesclous 2 pages.

pub mod runner;

pub use runner::{run, Tridesclous2Result, TRIDESCLOUS2_SORTER};

use crate::provenance::{Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};

/// Tridesclous 2 settings: SpikeInterface's `_default_params` and the defaults of the components it
/// calls (book: Tridesclous 2 parameters).
#[derive(Debug, Clone, PartialEq)]
pub struct Tridesclous2Config {
    pub do_bandpass: bool,
    pub bandpass_low_hz: f64,
    pub bandpass_high_hz: Option<f64>,
    pub filter_order: usize,
    pub do_common_reference: bool,
    pub common_reference_min_channels: usize,
    pub do_whiten: bool,
    pub whitening_radius_um: Option<f32>,
    pub whitening_chunks: usize,
    pub whitening_chunk_ms: f64,
    pub whitening_epsilon: f32,
    pub noise_chunks: usize,
    pub noise_chunk_ms: f64,
    pub detect_threshold: f64,
    pub detection_radius_um: f32,
    pub detection_exclude_sweep_ms: f64,
    pub n_peaks_per_channel: usize,
    pub min_n_peaks: usize,
    /// Clustering waveform window (ms).
    pub clustering_ms_before: f64,
    pub clustering_ms_after: f64,
    pub features_radius_um: f32,
    pub n_svd_components_per_channel: usize,
    pub svd_peaks_fit: usize,
    pub split_radius_um: f32,
    pub clustering_recursive_depth: usize,
    pub min_size_split: usize,
    pub n_pca_features: usize,
    pub isosplit_n_init: usize,
    pub isosplit_min_cluster_size: usize,
    pub isosplit_max_iterations_per_pass: usize,
    pub isocut_threshold: f64,
    pub clustering_sparsify_threshold: f64,
    pub clustering_min_snr: f64,
    pub merge_similarity: f64,
    pub merge_similarity_lag_ms: f64,
    pub min_firing_rate: f64,
    /// Matching templates' window (ms) and channels (µm around the peaks' barycentre).
    pub ms_before: f64,
    pub ms_after: f64,
    pub template_radius_um: f32,
    pub template_sparsify_threshold: f64,
    pub template_min_snr_ptp: f64,
    pub template_max_jitter_ms: f64,
    pub peeler_exclude_sweep_ms: f64,
    pub peeler_detection_radius_um: f32,
    pub peeler_cluster_radius_um: f32,
    pub peeler_amplitude_fitting_radius_um: f32,
    pub peeler_sample_shift: usize,
    pub peeler_ms_before: f64,
    pub peeler_ms_after: f64,
    pub peeler_max_loop: usize,
    pub peeler_amplitude_min: f64,
    pub peeler_amplitude_max: f64,
    pub peeler_fine_detector: bool,
    pub fine_detector_chunks: usize,
    pub final_merges: bool,
    pub final_merge_max_distance_um: f64,
    pub final_merge_censor_ms: f64,
    pub final_merge_sparsity_overlap: f64,
    pub chunk_sec: f64,
    pub seed: u64,
}

impl Default for Tridesclous2Config {
    fn default() -> Self {
        Self {
            do_bandpass: true,
            bandpass_low_hz: 150.0,
            bandpass_high_hz: Some(6000.0),
            filter_order: 2,
            do_common_reference: true,
            common_reference_min_channels: 32,
            do_whiten: true,
            whitening_radius_um: Some(100.0),
            whitening_chunks: 20,
            whitening_chunk_ms: 500.0,
            whitening_epsilon: 1e-16,
            noise_chunks: 20,
            noise_chunk_ms: 500.0,
            detect_threshold: 5.0,
            detection_radius_um: 150.0,
            detection_exclude_sweep_ms: 1.5,
            n_peaks_per_channel: 5000,
            min_n_peaks: 20_000,
            clustering_ms_before: 0.5,
            clustering_ms_after: 1.5,
            features_radius_um: 120.0,
            n_svd_components_per_channel: 5,
            svd_peaks_fit: 5000,
            split_radius_um: 60.0,
            clustering_recursive_depth: 3,
            min_size_split: 25,
            n_pca_features: 6,
            isosplit_n_init: 15,
            isosplit_min_cluster_size: 10,
            isosplit_max_iterations_per_pass: 500,
            isocut_threshold: 2.0,
            clustering_sparsify_threshold: 1.5,
            clustering_min_snr: 3.5,
            merge_similarity: 0.8,
            merge_similarity_lag_ms: 0.5,
            min_firing_rate: 0.1,
            ms_before: 1.0,
            ms_after: 2.5,
            template_radius_um: 100.0,
            template_sparsify_threshold: 1.5,
            template_min_snr_ptp: 3.5,
            template_max_jitter_ms: 0.2,
            peeler_exclude_sweep_ms: 0.8,
            peeler_detection_radius_um: 80.0,
            peeler_cluster_radius_um: 150.0,
            peeler_amplitude_fitting_radius_um: 150.0,
            peeler_sample_shift: 2,
            peeler_ms_before: 0.5,
            peeler_ms_after: 0.8,
            peeler_max_loop: 2,
            peeler_amplitude_min: 0.7,
            peeler_amplitude_max: 1.4,
            peeler_fine_detector: true,
            fine_detector_chunks: 5,
            final_merges: true,
            final_merge_max_distance_um: 50.0,
            final_merge_censor_ms: 3.0,
            final_merge_sparsity_overlap: 0.5,
            chunk_sec: 1.0,
            seed: 0,
        }
    }
}

/// Paper and code of Tridesclous 2.
pub fn tridesclous2_provenance() -> Provenance {
    Provenance {
        name: "Tridesclous 2".into(),
        kind: ProvenanceKind::PortedFromCode,
        paper: Some(Paper {
            title: "SpikeInterface, a unified framework for spike sorting".into(),
            authors: vec!["Buccino".into(), "Hurwitz".into(), "Garcia".into(), "Magland".into(), "Siegle".into(), "Hurwitz".into(), "Hennig".into()],
            venue: "eLife".into(),
            year: 2020,
            doi: "10.7554/eLife.61834".into(),
            license: None,
        }),
        code: UpstreamCode {
            url: "https://github.com/SpikeInterface/spikeinterface".into(),
            license: Some("MIT".into()),
            version: "sorter 2026.01 (main f08c987)".into(),
        },
        artifacts: Vec::new(),
        notes: "Ported from SpikeInterface's tridesclous2 and sortingcomponents (Tridesclous by Samuel Garcia and Christophe Pouzat has no dedicated paper; SpikeInterface is cited). Not yet: motion correction (off by default upstream). Upstream quirks kept are listed in the book.".into(),
    }
}

/// Tridesclous 2 over a whole recording ([`run`]).
#[derive(Debug, Clone, Default)]
pub struct Tridesclous2 {
    pub config: Tridesclous2Config,
}

impl Tridesclous2 {
    pub fn new(config: Tridesclous2Config) -> Self {
        Self { config }
    }

    pub fn run(
        &self,
        client: &cubecl::prelude::Client,
        source: &dyn dsp_core::RecordingSource,
        probe: &dsp_io::neuro::probe::SensorLayout,
    ) -> dsp_core::DspResult<Tridesclous2Result> {
        run(client, source, probe, &self.config, &dsp_core::NoProgress)
    }
}

impl Attributed for Tridesclous2 {
    fn provenance(&self) -> Provenance {
        tridesclous2_provenance()
    }
}
