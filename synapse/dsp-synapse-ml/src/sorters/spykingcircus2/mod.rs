//! SpyKING CIRCUS 2 (Yger et al., eLife 2018, for SpyKING CIRCUS; SpikeInterface's
//! `sorters/internal/spyking_circus2.py`, MIT, sorter version 2025.12), ported from its source:
//! Bessel band-pass, common median reference, local whitening, matched-filtering detection,
//! uniform peak selection, local SVD features, iterative HDBSCAN splits, template cleaning and
//! merging, circus-omp matching ([`super::components`]), driven by [`runner`].
//!
//! Not yet: motion correction (upstream's default on dense probes: S10). Stages and settings: the
//! book's SpyKING CIRCUS 2 pages.

pub mod runner;

pub use runner::{run, Spykingcircus2Result, SPYKINGCIRCUS2_SORTER};

use crate::provenance::{Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode};

/// SpyKING CIRCUS 2 settings: SpikeInterface's `_default_params` and the defaults of the components
/// it calls (book: SpyKING CIRCUS 2 parameters).
#[derive(Debug, Clone, PartialEq)]
pub struct Spykingcircus2Config {
    /// Waveform window around a peak (ms).
    pub ms_before: f64,
    pub ms_after: f64,
    /// Feature neighbourhood (µm); detection uses half of it.
    pub radius_um: f32,
    /// Bessel band-pass (a high-pass without an upper edge), forward-backward.
    pub do_bandpass: bool,
    pub bandpass_low_hz: f64,
    pub bandpass_high_hz: Option<f64>,
    pub filter_order: usize,
    /// Common median reference, on recordings of at least `common_reference_min_channels`.
    pub do_common_reference: bool,
    pub common_reference_min_channels: usize,
    pub do_whiten: bool,
    /// Local whitening over this radius (µm); `None`: global.
    pub whitening_radius_um: Option<f32>,
    pub whitening_chunks: usize,
    pub whitening_chunk_ms: f64,
    pub whitening_epsilon: f32,
    /// Noise levels: chunks and their length.
    pub noise_chunks: usize,
    pub noise_chunk_ms: f64,
    /// Detection threshold (× noise; matched filtering: × the filtered rows' noise).
    pub detect_threshold: f64,
    /// Peaks the detection prototype is the median of.
    pub prototype_peaks: usize,
    /// Chunks the matched filter's thresholds are fitted on.
    pub matched_filter_chunks: usize,
    /// Peaks clustered: `max(min_n_peaks, n_peaks_per_channel · channels)`.
    pub n_peaks_per_channel: usize,
    pub min_n_peaks: usize,
    pub svd_components: usize,
    pub svd_peaks_fit: usize,
    pub split_radius_um: f32,
    pub split_depth: usize,
    pub min_cluster_size: usize,
    pub split_pca_features: usize,
    pub sparsify_threshold: f64,
    pub min_snr: f64,
    pub max_jitter_ms: f64,
    pub mean_sd_ratio_threshold: f64,
    pub merge_similarity: f64,
    pub merge_num_shifts: usize,
    pub min_firing_rate: f64,
    pub omp_min_amplitude: f32,
    pub omp_max_failures: usize,
    pub omp_rank: usize,
    pub omp_vicinity: usize,
    /// Final cleaning (`auto_merge_units`, cross-contamination presets).
    pub final_merges: bool,
    /// Furthest apart (µm) two units' locations may be to merge.
    pub final_merge_max_distance_um: f64,
    /// Merged trains drop spikes closer than this (ms).
    pub final_merge_censor_ms: f64,
    /// Units merge only when their channels overlap (intersection / union) at least this much.
    pub final_merge_sparsity_overlap: f64,
    /// Template similarity lag each side (ms).
    pub final_merge_max_lag_ms: f64,
    /// Window length (s): SpikeInterface's chunk.
    pub chunk_sec: f64,
    pub seed: u64,
}

impl Default for Spykingcircus2Config {
    fn default() -> Self {
        Self {
            ms_before: 0.5,
            ms_after: 1.5,
            radius_um: 100.0,
            do_bandpass: true,
            bandpass_low_hz: 150.0,
            bandpass_high_hz: Some(7000.0),
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
            prototype_peaks: 10_000,
            matched_filter_chunks: 5,
            n_peaks_per_channel: 5000,
            min_n_peaks: 100_000,
            svd_components: 5,
            svd_peaks_fit: 5000,
            split_radius_um: 75.0,
            split_depth: 3,
            min_cluster_size: 20,
            split_pca_features: 3,
            sparsify_threshold: 1.0,
            min_snr: 5.0,
            max_jitter_ms: 0.2,
            mean_sd_ratio_threshold: 3.0,
            merge_similarity: 0.8,
            merge_num_shifts: 3,
            min_firing_rate: 0.1,
            omp_min_amplitude: 0.6,
            omp_max_failures: 5,
            omp_rank: 5,
            omp_vicinity: 2,
            final_merges: true,
            final_merge_max_distance_um: 50.0,
            final_merge_censor_ms: 3.0,
            final_merge_sparsity_overlap: 0.5,
            final_merge_max_lag_ms: 0.1,
            chunk_sec: 1.0,
            seed: 42,
        }
    }
}

/// Paper and code of SpyKING CIRCUS 2.
pub fn spykingcircus2_provenance() -> Provenance {
    Provenance {
        name: "SpyKING CIRCUS 2".into(),
        kind: ProvenanceKind::PortedFromCode,
        paper: Some(Paper {
            title: "A spike sorting toolbox for up to thousands of electrodes validated with ground truth recordings in vitro and in vivo".into(),
            authors: vec!["Yger".into(), "Spampinato".into(), "Esposito".into(), "Lefebvre".into(), "Deny".into(), "Gardella".into(), "Stimberg".into(), "Jetter".into(), "Zeck".into(), "Picaud".into(), "Duebel".into(), "Marre".into()],
            venue: "eLife".into(),
            year: 2018,
            doi: "10.7554/eLife.34518".into(),
            license: None,
        }),
        code: UpstreamCode {
            url: "https://github.com/SpikeInterface/spikeinterface".into(),
            license: Some("MIT".into()),
            version: "sorter 2025.12 (main f08c987)".into(),
        },
        artifacts: Vec::new(),
        notes: "Ported from SpikeInterface's spyking_circus2 and sortingcomponents, including the final auto_merge_units cleaning. Not yet: motion correction. Upstream quirks kept (template similarity fill, merge lag sign) are listed in the book.".into(),
    }
}

/// SpyKING CIRCUS 2 over a whole recording ([`run`]).
#[derive(Debug, Clone, Default)]
pub struct Spykingcircus2 {
    pub config: Spykingcircus2Config,
}

impl Spykingcircus2 {
    pub fn new(config: Spykingcircus2Config) -> Self {
        Self { config }
    }

    pub fn run(
        &self,
        client: &cubecl::prelude::Client,
        source: &dyn dsp_core::RecordingSource,
        probe: &dsp_io::neuro::probe::SensorLayout,
    ) -> dsp_core::DspResult<Spykingcircus2Result> {
        run(client, source, probe, &self.config, &dsp_core::NoProgress)
    }
}

impl Attributed for Spykingcircus2 {
    fn provenance(&self) -> Provenance {
        spykingcircus2_provenance()
    }
}
