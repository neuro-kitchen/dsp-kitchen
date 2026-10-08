//! Neural spike detection, spatial deduplication, waveform alignment, feature extraction,
//! localization, drift estimation, kriging, clustering, template matching, correlograms,
//! Allen/IBL quality metrics, sorter comparison, and neural frequency bands.
//!
//! Probe geometry (layouts, presets, nearest-site queries) lives in `dsp_io::neuro::probe`.
//!
//! # Where things are
//!
//! | Step | Module | Start with |
//! |---|---|---|
//! | Find spikes | [`detection`] | [`detect_spikes_multichannel`], [`deduplicate_spikes_spatial`] |
//! | Whole recordings, bounded memory | [`streaming`] | [`streaming::StreamingDetector`] |
//! | Cut waveforms | [`extraction`] | [`extract_snippets_multichannel`] |
//! | Features | [`features`] | [`PcaFeatureEmbedder`] |
//! | Positions and drift | [`spatial`] | [`PeakLocalizer`] implementations |
//! | Cluster into units | [`sorting`] | k-means, GMM, HDBSCAN, bipartite modularity |
//! | Quality and comparison | [`metrics`] | ISI violations, presence, amplitude cutoff; sorter agreement |
//! | Results on disk | [`storage`] | [`SortingOutput`] (Phy, Zarr, NWB) |
//!
//! Complete sorters (Kilosort4, EMUsort) are in `dsp-synapse-ml`.
//!
//! # Example
//!
//! Detect threshold crossings, then keep one spike per event: a spike seen on neighbouring
//! channels is one spike, assigned to the channel where it is largest.
//!
//! ```
//! use dsp_io::neuro::probe::{Position3D, SensorLayout, SensorSite};
//! use dsp_synapse::{deduplicate_spikes_spatial, detect_spikes_multichannel, SpikePolarity, SpikeSpacing};
//!
//! // 4 channels on a line, 20 µm apart
//! let sites = (0..4).map(|c| SensorSite::new(c, Position3D::new(0.0, 20.0 * c as f32, 0.0), 0)).collect();
//! let layout = SensorLayout::new("linear", sites);
//!
//! // Small noise, and one spike at sample 500: −80 µV on channel 1, −40 µV on its neighbour
//! let (channels, samples) = (4, 1_000);
//! let mut x: Vec<f32> = (0..channels * samples).map(|i| ((i * 7_919) % 200) as f32 / 100.0 - 1.0).collect();
//! x[samples + 500] = -80.0;
//! x[2 * samples + 500] = -40.0;
//!
//! // Troughs below −5 σ (σ per channel, from the median absolute deviation), ≥ 1 ms apart at 30 kHz
//! let crossings = detect_spikes_multichannel(&x, channels, samples, 5.0, SpikePolarity::Negative, SpikeSpacing::new(30));
//! assert_eq!(crossings.len(), 2); // the same spike, on two channels
//!
//! // Crossings within 50 µm and 10 samples of a larger one belong to it
//! let spikes = deduplicate_spikes_spatial(&crossings, &layout, 50.0, 10);
//! assert_eq!(spikes.len(), 1);
//! assert_eq!(spikes[0].primary_channel, 1);
//! assert_eq!(spikes[0].participating_channels, vec![1, 2]);
//! ```

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
    DedupNeighbours, compute_neo_energy_1d, deduplicate_spikes_spatial, deduplicate_spikes_spatial_gpu,
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
    KdeMergeResult, MotorUnitPulseTrain, MatchingPursuitMatcher, cluster_density_peaks,
    cluster_density_peaks_capped, cluster_gmm_bic, cluster_kde_merge,
    compute_template_similarity_matrix, match_spikes_matching_pursuit,
    suggest_template_merges, template_max_cosine_similarity,
};
pub use spatial::{
    CenterOfMassLocalizer, DipoleEstimate, DipoleLocalizer, DriftEstimate, GridConvolutionLocalizer,
    KRIGING_REGULARIZATION, MonopolarTriangulator, NonRigidDriftEstimate, TraceKriging,
    compute_kriging_weight_matrix,
    correct_snippet_batch_drift_kriging, correct_traces_drift_kriging, estimate_nonrigid_drift,
    estimate_rigid_drift, localize_spike_center_of_mass, localize_spike_dipole,
    localize_spike_grid_convolution, localize_spike_monopolar,
};
pub use streaming::{
    BatchTemplateStats, SINC_RESAMPLE_MARGIN, StreamingDetectionConfig, StreamingDetectionResult,
    StreamingDetector, TemplateAccumulator, execute_reduce_templates_in_vram,
    reduce_channel_templates_kernel,
};
pub use storage::{
    fill_similarity, load_nwb_units, load_phy_folder, load_sorting,
    load_sorting_zarr, load_spikes, save_nwb_units, save_phy_folder, save_sorting,
    save_sorting_zarr,
};
