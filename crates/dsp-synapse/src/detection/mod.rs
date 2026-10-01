pub mod dedup;
pub mod kernels;
pub mod matched_filter;
pub mod neo;
pub mod noise;
pub mod threshold;

pub use crate::core::{DeduplicatedSpike, SpikeEvent};
pub use dedup::{StreamingDedup, deduplicate_spikes_spatial};
pub use kernels::{
    DetectionCarry, count_trough_candidates_kernel, execute_detect_spikes_in_vram,
    scan_candidate_counts_kernel, write_trough_candidates_kernel,
};
pub use matched_filter::{
    MatchedFilterSpikeDetector, canonical_biphasic_prototype, detect_spikes_matched_filter,
};
pub use neo::{NeoSpikeDetector, compute_neo_energy_1d, detect_spikes_neo};
pub use noise::estimate_noise_std;
pub use threshold::{
    ThresholdSpikeDetector, detect_spikes_multichannel, detect_spikes_with_sigma,
};
