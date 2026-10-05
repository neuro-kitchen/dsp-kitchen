pub mod cbss;
pub mod density_peaks;
pub mod gmm;
pub mod hdbscan;
pub mod kde_merge;
pub mod kmeans;
pub mod kernels;
pub mod matching_pursuit;
pub mod similarity;

pub use cbss::{ConvolutiveBssDecomposer, MotorUnitPulseTrain};
pub use density_peaks::{DensityPeaksResult, cluster_density_peaks, cluster_density_peaks_capped};
pub use gmm::{GmmClusterer, GmmCovarianceKind, GmmResult, cluster_gmm_bic};
pub use hdbscan::hdbscan;
pub use kde_merge::{KdeMergeResult, cluster_kde_merge};
pub use kmeans::{KMeansOptions, KMeansResult, kmeans};
pub use kernels::{mp_score_kernel, mp_subtract_kernel};
pub use matching_pursuit::{MatchingPursuitMatcher, match_spikes_matching_pursuit};
pub use similarity::{
    compute_template_similarity_matrix, suggest_template_merges, template_max_cosine_similarity,
};
