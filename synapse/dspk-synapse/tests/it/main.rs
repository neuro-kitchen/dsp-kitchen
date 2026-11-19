//! dspk-synapse integration tests, one binary (linked once): each file a module.
//!
//! Run one module: `cargo test --test it -- <module>::`.

mod extraction_alignment;
mod quality_metrics_spikeinterface;
mod streaming_invariance;
