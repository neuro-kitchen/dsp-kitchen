pub mod density_peaks;
pub mod kernels;
pub mod omp;
pub mod similarity;

pub use density_peaks::{DensityPeaksResult, cluster_density_peaks, cluster_density_peaks_capped};
pub use kernels::{omp_score_kernel, omp_subtract_kernel};
pub use omp::{OmpSpikeMatcher, match_spikes_omp, match_spikes_omp_on};
pub use similarity::{
    compute_template_similarity_matrix, suggest_template_merges, template_max_cosine_similarity,
};
