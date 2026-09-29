//! Neural spike detection, spatial deduplication, waveform alignment, PCA extraction, neural frequency bands, quality metrics, and probe geometries.

pub mod bands;
pub mod probe;
pub mod detection;
pub mod extraction;
pub mod features;
pub mod metrics;

pub use bands::NeuralBand;
pub use detection::{
    SpikeEvent, DeduplicatedSpike, detect_spikes_multichannel, estimate_noise_std,
    deduplicate_spikes_spatial,
};
pub use extraction::{
    WaveformSnippet, extract_snippets_multichannel, extract_snippets_single_channel,
    parabolic_subsample_offset, resample_sinc_1d, resample_sinc_multichannel,
};
pub use features::{SpikeMorphology, compute_morphology, extract_waveform_pca};
pub use metrics::{
    IsiMetrics, WaveformTemplate, compute_isi_violations, compute_snr, compute_mean_template,
};
pub use probe::{neuropixels_1_0, neuropixels_2_0, tetrode, utah_array, find_k_nearest_neighbors};
