//! Spike detection: per-channel candidates from dspk-ops peak finding (`dspk_ops::peaks`), accepted
//! by each detector's score and spaced by [`SpikeSpacing`]; then spatial deduplication.

pub mod adaptive;
pub mod dedup;
mod device;
pub(crate) mod kernels;
pub mod matched_filter;
pub mod neo;
pub mod noise;
mod spacing;
pub mod threshold;

pub use adaptive::AdaptiveThresholdDetector;
pub use dedup::{
    DedupNeighbours, DeduplicatedSpike, StreamingDedup, deduplicate_spikes_spatial, deduplicate_spikes_spatial_device,
};
pub use device::execute_detect_spikes_in_vram;
pub use dspk_ops::peaks::DistanceRule;
pub use matched_filter::{
    MatchedFilterSpikeDetector, canonical_biphasic_prototype, detect_spikes_matched_filter,
};
pub use neo::{NeoSpikeDetector, compute_neo_energy, detect_spikes_neo};
pub use noise::{
    estimate_noise_rms, estimate_noise_std, estimate_noise_trimmed, interquartile_range,
    standard_error,
};
pub use spacing::SpikeSpacing;
pub use threshold::{
    SpikeEvent, SpikePolarity, ThresholdSpikeDetector, detect_spikes_threshold,
    detect_spikes_threshold_with_sigma, detection_heights,
};
