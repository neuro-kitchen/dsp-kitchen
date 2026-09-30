//! Neural spike detection, spatial deduplication, waveform alignment, PCA extraction,
//! localization, drift estimation, kriging, clustering, template matching, correlograms,
//! Allen/IBL quality metrics, neural frequency bands, and probe geometries.

pub mod bands;
pub mod clustering;
pub mod correlogram;
pub mod detection;
pub mod extraction;
pub mod features;
pub mod kernels;
pub mod localization;
pub mod matching;
pub mod metrics;
pub mod motion;
pub mod probe;
pub mod streaming;
pub mod traits;

pub use bands::NeuralBand;
pub use clustering::{DensityPeaksResult, cluster_density_peaks, cluster_density_peaks_capped};
pub use correlogram::{Correlogram, compute_autocorrelogram, compute_crosscorrelogram};
pub use detection::{
    SpikeEvent, DeduplicatedSpike, ThresholdSpikeDetector, NeoSpikeDetector,
    MatchedFilterSpikeDetector, detect_spikes_multichannel, detect_spikes_with_sigma,
    detect_spikes_neo, detect_spikes_matched_filter, compute_neo_energy_1d,
    canonical_biphasic_prototype, estimate_noise_std, deduplicate_spikes_spatial,
};
pub use extraction::{
    WaveformSnippet, SnippetBatch, extract_snippets_multichannel, extract_snippets_single_channel,
    extract_snippet_batch_multichannel, parabolic_subsample_offset, resample_sinc_1d,
    resample_sinc_multichannel,
};
pub use features::{SpikeMorphology, PcaFeatureEmbedder, compute_morphology, extract_waveform_pca};
pub use kernels::{
    BatchTemplateStats, detect_channel_troughs_kernel, execute_detect_spikes_in_vram,
    extract_sinc_snippets_kernel, execute_extract_sinc_in_vram, reduce_channel_templates_kernel,
    execute_reduce_templates_in_vram,
};
pub use localization::{
    CenterOfMassLocalizer, MonopolarTriangulator, GridConvolutionLocalizer,
    localize_spike_center_of_mass, localize_spike_monopolar, localize_spike_grid_convolution,
};
pub use matching::{
    OmpSpikeMatcher, match_spikes_omp, template_max_cosine_similarity,
    compute_template_similarity_matrix, suggest_template_merges,
};
pub use metrics::{
    IsiMetrics, WaveformTemplate, compute_isi_violations, compute_snr, compute_mean_template,
    compute_amplitude_cutoff, compute_amplitude_cutoff_with, compute_presence_ratio, count_refractory_violations,
    compute_llobet_contamination, compute_d_prime, compute_silhouette_score,
    compute_isolation_distance,
};
pub use motion::{
    DriftEstimate, estimate_rigid_drift,
    compute_kriging_weight_matrix, correct_snippet_batch_drift_kriging, correct_traces_drift_kriging,
};
pub use probe::{
    neuropixels_1_0, neuropixels_2_0, tetrode, utah_array, find_k_nearest_neighbors,
    precompute_knn_table,
};
pub use streaming::{
    SINC_RESAMPLE_MARGIN, StreamingSortConfig, StreamingSortResult, StreamingSpikeRunner,
    TemplateAccumulator,
};
pub use traits::{
    SpikeDetector, WaveformDenoiser, FeatureEmbedder, PeakLocalizer, SpikeMatcher,
    MatchedSpike, UnitQualityLabel,
};
