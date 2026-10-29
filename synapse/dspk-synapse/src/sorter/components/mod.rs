//! Building blocks shared by the SpikeInterface-based sorters (SpyKING CIRCUS 2, Tridesclous 2),
//! ported from SpikeInterface's `sortingcomponents` (MIT).

pub mod clustering;
pub mod detection;
pub mod matching;
pub mod templates;

pub use clustering::{auto_merge, split_clusters, truncated_svd, AutoMergeOptions, AutoMerged, SparseFeatures, SplitOptions};
pub use detection::{
    convolution_weights, gather_points, locally_exclusive, Candidate, LocallyExclusiveDetector, MatchedFilter, MatchedFilterOptions,
    MatchedPeak, Peak, PeakSign,
};
pub use matching::{fine_prototype, CircusOmp, FineFilter, OmpOptions, OmpSpike, TdcPeeler, TdcPeelerOptions, TdcSpike};
pub use templates::{
    clean_templates, merge_by_similarity, prototype, remove_small_clusters, template_similarity, templates_from_svd, CleanOptions,
    Merged, Templates,
};
