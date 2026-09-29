//! Neural spike detection, waveform alignment, PCA extraction, neural frequency bands, and probe geometries.

pub mod bands;
pub mod probe;
pub mod detection;
pub mod extraction;
pub mod features;

pub use bands::NeuralBand;
pub use detection::{SpikeEvent, detect_spikes_multichannel, estimate_noise_std};
pub use extraction::{WaveformSnippet, extract_snippets_single_channel, parabolic_subsample_offset};
pub use features::{SpikeMorphology, compute_morphology, extract_waveform_pca};
pub use probe::{neuropixels_1_0, neuropixels_2_0, tetrode, utah_array, find_k_nearest_neighbors};
