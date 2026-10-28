//! dspk: multi-channel signal processing written once in CubeCL, run on any GPU (wgpu, CUDA, HIP)
//! or the CPU, with the device chosen at run time.
//!
//! This crate re-exports the dspk crates; features choose which are included and which runtimes
//! are compiled in (default: `wgpu`, `zarr`).
//!
//! | Module | Crate | Contents |
//! |---|---|---|
//! | [`core`] | `dspk-core` | recordings, buffers, time, windows, compute targets |
//! | [`io`] | `dspk-io` | file formats (raw binary, Zarr), detection, chunk cache |
//! | [`ops`] | `dspk-ops` | filters, resampling, linear algebra, statistics, peaks, spatial operators |
//! | `view` (feature `view`) | `dspk-view` | min/max envelopes and pyramids |
//! | `stream` (feature `stream`) | `dspk-stream` | QUIC + TLS sessions for continuous signals |
//!
//! Early development (0.0.x): APIs change between releases.

pub use dspk_core as core;
pub use dspk_io as io;
pub use dspk_ops as ops;
#[cfg(feature = "stream")]
pub use dspk_stream as stream;
#[cfg(feature = "view")]
pub use dspk_view as view;
