pub mod dedup;
pub mod matched_filter;
pub mod neo;
pub mod noise;
pub mod threshold;

pub use dedup::{DeduplicatedSpike, deduplicate_spikes_spatial};
pub use matched_filter::{
    MatchedFilterSpikeDetector, canonical_biphasic_prototype, detect_spikes_matched_filter,
};
pub use neo::{NeoSpikeDetector, compute_neo_energy_1d, detect_spikes_neo};
pub use noise::estimate_noise_std;
pub use threshold::{
    SpikeEvent, ThresholdSpikeDetector, detect_spikes_multichannel, detect_spikes_with_sigma,
};
