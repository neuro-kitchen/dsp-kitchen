pub mod noise;
pub mod threshold;

pub use noise::estimate_noise_std;
pub use threshold::{SpikeEvent, detect_spikes_multichannel};
