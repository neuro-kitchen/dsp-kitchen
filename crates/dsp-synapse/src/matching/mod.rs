pub mod omp;
pub mod similarity;

pub use omp::{OmpSpikeMatcher, match_spikes_omp, match_spikes_omp_on};
pub use similarity::{
    compute_template_similarity_matrix, suggest_template_merges, template_max_cosine_similarity,
};
