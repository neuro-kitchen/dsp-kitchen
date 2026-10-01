//! Neural spike detection, spatial deduplication, waveform alignment, PCA extraction,
//! localization, drift estimation, kriging, clustering, template matching, correlograms,
//! Allen/IBL quality metrics, neural frequency bands, and probe geometries.

pub mod core;
pub mod detection;
pub mod extraction;
pub mod features;
pub mod metrics;
pub mod probe;
pub mod sorting;
pub mod spatial;
pub mod streaming;

// Compatibility module aliases so downstream callers referencing `dsp_synapse::{bands, clustering, correlogram, kernels, localization, matching, motion, traits}` continue to work seamlessly.
pub use core::{bands, traits};
pub use metrics::correlogram;
pub use sorting as clustering;
pub use sorting as matching;
pub use spatial as localization;
pub use spatial as motion;

pub mod kernels {
    pub use crate::detection::kernels::*;
    pub use crate::extraction::kernels::*;
    pub use crate::sorting::kernels::*;
    pub use crate::streaming::kernels::*;
}

pub use core::{
    DeduplicatedSpike, FeatureEmbedder, MatchedSpike, NeuralBand, PeakLocalizer, SnippetBatch,
    SpikeDetector, SpikeEvent, SpikeMatcher, UnitQualityLabel, WaveformDenoiser, WaveformSnippet,
    WaveformTemplate, compute_mean_template,
};
pub use detection::{
    AdaptiveThresholdDetector, MatchedFilterSpikeDetector, NeoSpikeDetector, SpikePolarity,
    ThresholdSpikeDetector, canonical_biphasic_prototype, compute_neo_energy_1d,
    count_trough_candidates_kernel, deduplicate_spikes_spatial, deduplicate_spikes_spatial_gpu,
    detect_spikes_matched_filter, detect_spikes_multichannel, detect_spikes_multichannel_polarity,
    detect_spikes_neo, detect_spikes_with_sigma, detect_spikes_with_sigma_polarity,
    estimate_noise_std, execute_detect_spikes_in_vram,
};
pub use extraction::{
    execute_extract_sinc_in_vram, extract_sinc_snippets_kernel,
    extract_snippet_batch_multichannel, extract_snippets_multichannel,
    extract_snippets_single_channel, parabolic_subsample_offset, resample_sinc_1d,
    resample_sinc_multichannel,
};
pub use features::{
    ConductionVelocityEstimate, PcaFeatureEmbedder, PpcaFeatureEmbedder, SpikeMorphology,
    WaveletFeatureEmbedder, compute_morphology, estimate_hdemg_conduction_velocity,
    extract_waveform_pca, haar_dwt_multilevel_1d,
};
pub use metrics::{
    Correlogram, IsiMetrics, compute_amplitude_cutoff, compute_amplitude_cutoff_with,
    compute_autocorrelogram, compute_crosscorrelogram, compute_d_prime,
    compute_isi_violations, compute_isolation_distance, compute_llobet_contamination,
    compute_presence_ratio, compute_silhouette_score, compute_snr, count_refractory_violations,
};
pub use probe::{
    find_k_nearest_neighbors, hdemg_4x8, hdemg_8x8, hdemg_grid, neuropixels_1_0, neuropixels_2_0,
    precompute_knn_table, tetrode, utah_array,
};
pub use sorting::{
    ConvolutiveBssDecomposer, DensityPeaksResult, GmmClusterer, GmmCovarianceKind, GmmResult,
    IsoSplitResult, MotorUnitPulseTrain, OmpSpikeMatcher, cluster_density_peaks,
    cluster_density_peaks_capped, cluster_gmm_bic, cluster_isosplit,
    compute_template_similarity_matrix, match_spikes_omp, match_spikes_omp_on,
    suggest_template_merges, template_max_cosine_similarity,
};
pub use spatial::{
    CenterOfMassLocalizer, DipoleEstimate, DipoleLocalizer, DriftEstimate, GridConvolutionLocalizer,
    MonopolarTriangulator, NonRigidDriftEstimate, compute_kriging_weight_matrix,
    correct_snippet_batch_drift_kriging, correct_traces_drift_kriging, estimate_nonrigid_drift,
    estimate_rigid_drift, localize_spike_center_of_mass, localize_spike_dipole,
    localize_spike_grid_convolution, localize_spike_monopolar,
};
pub use streaming::{
    BatchTemplateStats, SINC_RESAMPLE_MARGIN, StreamingSortConfig, StreamingSortResult,
    StreamingSpikeRunner, TemplateAccumulator, execute_reduce_templates_in_vram,
    reduce_channel_templates_kernel,
};
