//! Kernel benchmarks (`--ignored`), one test binary: each file a module.
//!
//! Run one module: `cargo test --test bench -- <module>::`.

mod bench_eigen;
mod bench_hdbscan;
mod bench_kernels;
mod bench_linalg;
mod bench_reduce;
mod bench_upload;
mod bench_vector;
