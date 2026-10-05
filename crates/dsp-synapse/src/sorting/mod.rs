pub mod cbss;
pub mod density_peaks;
pub mod gmm;
pub mod isosplit;
pub mod kernels;
pub mod omp;
pub mod similarity;

pub use cbss::{ConvolutiveBssDecomposer, MotorUnitPulseTrain};
pub use density_peaks::{DensityPeaksResult, cluster_density_peaks, cluster_density_peaks_capped};
pub use gmm::{GmmClusterer, GmmCovarianceKind, GmmResult, cluster_gmm_bic};
pub use isosplit::{IsoSplitResult, cluster_isosplit};
pub use kernels::{omp_score_kernel, omp_subtract_kernel};
pub use omp::{OmpSpikeMatcher, match_spikes_omp, match_spikes_omp_on};
pub use similarity::{
    compute_template_similarity_matrix, suggest_template_merges, template_max_cosine_similarity,
};
