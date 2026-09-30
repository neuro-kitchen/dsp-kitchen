pub mod probe;
pub mod detection;
pub mod extraction;
pub mod metrics;
pub mod streaming;

pub use probe::PyProbeLayout;
pub use detection::{PySpikeEvent, PyDeduplicatedSpike, detect_spikes, deduplicate_spikes, estimate_noise};
pub use extraction::{PyWaveformSnippet, extract_snippets};
pub use metrics::{compute_isi, compute_snr, compute_template};
pub use streaming::{PyStreamingSortResult, sort_recording};
