//! Building blocks shared by the SpikeInterface-based sorters (SpyKING CIRCUS 2, Tridesclous 2),
//! ported from SpikeInterface's `sortingcomponents` (MIT).

pub mod automerge;
pub mod detect;
pub mod exclusive;
pub mod kernels;
pub mod matched;
pub mod omp;
pub mod prototype;
pub mod split;
pub mod tdc_peeler;
pub mod templates;

pub use automerge::{auto_merge, AutoMergeOptions, AutoMerged};
pub use detect::{LocallyExclusiveDetector, Peak, PeakSign};
pub use exclusive::{locally_exclusive, Candidate};
pub use omp::{CircusOmp, OmpOptions, OmpSpike};
pub use prototype::prototype;
pub use templates::{clean_templates, merge_by_similarity, remove_small_clusters, template_similarity, templates_from_svd, CleanOptions, Merged, Templates};
pub use tdc_peeler::{fine_prototype, FineFilter, TdcPeeler, TdcPeelerOptions, TdcSpike};
pub use split::{split_clusters, truncated_svd, SparseFeatures, SplitOptions};
pub use matched::{convolution_weights, gather_points, MatchedFilter, MatchedFilterOptions, MatchedPeak};
