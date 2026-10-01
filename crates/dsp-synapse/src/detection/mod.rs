pub mod adaptive;
pub mod dedup;
pub mod kernels;
pub mod matched_filter;
pub mod neo;
pub mod noise;
pub mod threshold;

pub use adaptive::AdaptiveThresholdDetector;
pub use dedup::{
    DeduplicatedSpike, StreamingDedup, deduplicate_spikes_spatial, deduplicate_spikes_spatial_gpu,
};
pub use kernels::{
    DetectionCarry, count_trough_candidates_kernel, execute_detect_spikes_in_vram,
    scan_candidate_counts_kernel, spatial_dedup_survival_kernel, write_trough_candidates_kernel,
};
pub use matched_filter::{
    MatchedFilterSpikeDetector, canonical_biphasic_prototype, detect_spikes_matched_filter,
};
pub use neo::{NeoSpikeDetector, compute_neo_energy_1d, detect_spikes_neo};
pub use noise::{
    estimate_noise_rms, estimate_noise_std, estimate_noise_trimmed, interquartile_range,
    standard_error,
};
pub use threshold::{
    SpikeEvent, SpikePolarity, ThresholdSpikeDetector, detect_spikes_multichannel,
    detect_spikes_multichannel_polarity, detect_spikes_with_sigma,
    detect_spikes_with_sigma_polarity,
};
