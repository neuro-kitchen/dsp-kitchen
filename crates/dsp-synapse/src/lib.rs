//! Neural spike detection, spatial deduplication, waveform alignment, feature extraction,
//! localization, drift estimation, kriging, clustering, template matching, correlograms,
//! Allen/IBL quality metrics, sorter comparison, and neural frequency bands.
//!
//! Probe geometry (layouts, presets, nearest-site queries) lives in `dsp_io::neuro::probe`.

pub mod core;
pub mod detection;
pub mod extraction;
pub mod features;
pub mod metrics;
pub mod sorting;
pub mod spatial;
pub mod storage;
pub mod streaming;

pub use core::{
    DeduplicatedSpike, FeatureEmbedder, MatchedSpike, NeuralBand, PeakLocalizer, RecordingMeta,
    SnippetBatch, SortedUnit, SortingOutput, SpikeDetector, SpikeEvent, SpikeMatcher,
    UnitQualityLabel, WaveformDenoiser, WaveformSnippet, WaveformTemplate, compute_mean_template,
    dense_waveform, pack_templates, unpack_template,
};
pub use detection::{
    AdaptiveThresholdDetector, DistanceRule, MatchedFilterSpikeDetector, NeoSpikeDetector,
    SpikePolarity, SpikeSpacing, ThresholdSpikeDetector, canonical_biphasic_prototype,
    compute_neo_energy_1d, deduplicate_spikes_spatial, deduplicate_spikes_spatial_gpu,
    detect_spikes_matched_filter, detect_spikes_multichannel, detect_spikes_neo,
    detect_spikes_with_sigma, detection_heights, estimate_noise_std, execute_detect_spikes_in_vram,
};
pub use extraction::{
    execute_extract_sinc_in_vram, extract_snippet_batch_multichannel, extract_snippets_kernel,
    extract_snippets_multichannel, extract_snippets_single_channel, read_snippets,
    trough_shift_kernel,
};
pub use features::{
    ConductionVelocityEstimate, PcaFeatureEmbedder, PpcaFeatureEmbedder, SpikeMorphology,
    WaveletFeatureEmbedder, compute_morphology, estimate_hdemg_conduction_velocity,
    extract_waveform_pca, haar_dwt_multilevel_1d,
};
pub use metrics::{
    BurstEpoch, Correlogram, FiringRateCurve, IsiMetrics, MepMetrics, PairwiseTrainMatch,
    PsthResult, QualityCriteria, SortingComparison, StimulusTriggeredAverage, UnitMatchSummary,
    compare_sortings, compare_spike_trains, compute_amplitude_cutoff,
    compute_amplitude_cutoff_with, compute_autocorrelogram, compute_crosscorrelogram,
    compute_d_prime, compute_instantaneous_firing_rate, compute_isi_violations,
    compute_isolation_distance, compute_llobet_contamination, compute_presence_ratio,
    compute_psth, compute_silhouette_score, compute_snr, compute_stimulus_triggered_average,
    count_refractory_violations, detect_burst_epochs, isi_histogram, quantify_mep,
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
pub use storage::{
    fill_similarity, load_nwb_units, load_phy_folder, load_sorting,
    load_sorting_zarr, load_spikes, save_nwb_units, save_phy_folder, save_sorting,
    save_sorting_zarr,
};
