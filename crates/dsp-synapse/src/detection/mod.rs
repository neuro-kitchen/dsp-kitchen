//! Spike detection: per-channel candidates from dsp-base peak finding (`dsp_base::peaks`), accepted
//! by each detector's score and spaced by [`SpikeSpacing`]; then spatial deduplication.

pub mod adaptive;
pub mod dedup;
mod device;
pub mod kernels;
pub mod matched_filter;
pub mod neo;
pub mod noise;
mod spacing;
pub mod threshold;

pub use adaptive::AdaptiveThresholdDetector;
pub use dedup::{
    DeduplicatedSpike, StreamingDedup, deduplicate_spikes_spatial, deduplicate_spikes_spatial_gpu,
};
pub use device::execute_detect_spikes_in_vram;
pub use dsp_base::peaks::DistanceRule;
pub use kernels::spatial_dedup_survival_kernel;
pub use matched_filter::{
    MatchedFilterSpikeDetector, canonical_biphasic_prototype, detect_spikes_matched_filter,
};
pub use neo::{NeoSpikeDetector, compute_neo_energy_1d, detect_spikes_neo};
pub use noise::{
    estimate_noise_rms, estimate_noise_std, estimate_noise_trimmed, interquartile_range,
    standard_error,
};
pub use spacing::SpikeSpacing;
pub use threshold::{
    SpikeEvent, SpikePolarity, ThresholdSpikeDetector, detect_spikes_multichannel,
    detect_spikes_with_sigma, detection_heights,
};
