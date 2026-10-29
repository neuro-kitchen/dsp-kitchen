//! Cluster refinement: iterative splits on sparse features, and template-based auto-merging.

pub mod automerge;
pub mod split;

pub use automerge::{auto_merge, AutoMergeOptions, AutoMerged};
pub use split::{split_clusters, truncated_svd, SparseFeatures, SplitOptions};
