pub mod noise;
pub mod threshold;
pub mod dedup;

pub use noise::estimate_noise_std;
pub use threshold::{SpikeEvent, detect_spikes_multichannel};
pub use dedup::{DeduplicatedSpike, deduplicate_spikes_spatial};
