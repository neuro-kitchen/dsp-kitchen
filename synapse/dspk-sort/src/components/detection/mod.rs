//! Spike detection: locally exclusive peaks (SpikeInterface `locally_exclusive`) and matched
//! filtering with a prototype waveform (`matched_filtering`).

pub mod detect;
pub mod exclusive;
pub mod kernels;
pub mod matched;

pub use detect::{LocallyExclusiveDetector, Peak, PeakSign};
pub use exclusive::{locally_exclusive, Candidate};
pub use matched::{convolution_weights, gather_points, MatchedFilter, MatchedFilterOptions, MatchedPeak};
