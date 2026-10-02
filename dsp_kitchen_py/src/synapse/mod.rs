pub mod detection;
pub mod extraction;
pub mod metrics;
pub mod probe;
pub mod sorting;
pub mod spatial;
pub mod streaming;

pub use detection::{
    deduplicate_spikes, detect_spikes, estimate_noise, PyDeduplicatedSpike, PySpikeEvent,
};
pub use extraction::{extract_snippets, PyWaveformSnippet};
pub use metrics::{
    compute_amplitude_cutoff, compute_autocorrelogram, compute_crosscorrelogram, compute_d_prime,
    compute_firing_rate, compute_isi, compute_isolation_distance, compute_presence_ratio,
    compute_psth, compute_silhouette_score, compute_snr, compute_sta, compute_template,
    quantify_mep,
};
pub use probe::PyProbeLayout;
pub use sorting::{
    cluster_density_peaks, cluster_gmm, cluster_isosplit, decompose_hdemg_cbss, match_spikes_omp,
};
pub use spatial::{
    correct_drift_kriging, estimate_nonrigid_drift, estimate_rigid_drift, localize_spikes,
};
pub use streaming::{sort_recording, PyStreamingSortResult};
