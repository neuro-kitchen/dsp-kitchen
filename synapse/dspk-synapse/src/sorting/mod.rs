pub mod bipartite;
pub mod cbss;
pub mod density_peaks;
pub mod gmm;
pub mod hdbscan;
pub mod kde_merge;
pub mod kmeans;
/// Device kernels; public only for the sorter crates that launch them (`relabel_kernel`).
#[doc(hidden)]
pub mod kernels;
pub mod matching_pursuit;
pub mod points;
pub mod similarity;
pub mod subsample;

pub use bipartite::{execute_bipartite_clustering, execute_nearest_neighbours, BipartiteClustering, BipartiteOptions};
pub use cbss::{ConvolutiveBssDecomposer, MotorUnitPulseTrain};
pub use density_peaks::{DensityPeaksOptions, DensityPeaksResult, cluster_density_peaks};
pub use gmm::{GmmClusterer, GmmCovarianceKind, GmmResult, cluster_gmm_bic};
pub use hdbscan::{HdbscanOptions, DEFAULT_MIN_CLUSTER_SIZE, cluster_hdbscan, execute_hdbscan, execute_hdbscan_with_progress, cluster_hdbscan_progress_total};
pub use points::DevicePoints;
pub use subsample::{subsample_peaks, SubsampleOptions};
pub use kde_merge::{KdeMergeOptions, KdeMergeResult, cluster_kde_merge, DEFAULT_DIP_THRESHOLD, DEFAULT_KDE_MIN_CLUSTER_SIZE};
pub use kmeans::{KMeansOptions, KMeansResult, execute_cluster_sums, cluster_kmeans, execute_kmeans, execute_kmeans_with_progress};
pub use matching_pursuit::{MatchingPursuitMatcher, match_spikes_matching_pursuit};
pub use similarity::{
    compute_template_similarity_matrix, suggest_template_merges, template_max_cosine_similarity,
};
