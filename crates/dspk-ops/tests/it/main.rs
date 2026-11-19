//! dspk-ops integration tests, one binary (linked once): each file a module. `pipeline_fused` stays a binary of its own: a captured graph records every launch the process makes.
//!
//! Run one module: `cargo test --test it -- <module>::`.

mod accuracy;
mod common;
mod filter_blocks;
mod filter_chunks;
mod filter_runtimes;
mod filter_scipy;
mod kernel_runtimes;
mod properties;
