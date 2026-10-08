pub mod cluster;
pub mod detect;
pub mod learned;
pub mod matching;

pub use detect::{centre_response_kernel, correlate_templates_kernel, local_peak_score_kernel, neighbour_max_kernel, spike_features_kernel};
