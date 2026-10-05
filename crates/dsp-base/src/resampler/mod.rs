//! Sample-rate conversion on the device, following `scipy.signal`:
//!
//! - [`resample_poly`]: rational `up / down` resampling with a polyphase FIR (`resample_poly`).
//! - [`decimate`]: integer down-sampling behind an anti-aliasing filter (`decimate`).
//! - [`design`]: the windowed-sinc FIR design both use (`firwin`).

pub mod decimate;
pub mod design;
pub mod kernels;
pub mod poly;

pub use decimate::{decimate, decimate_len, DecimateFilter, DECIMATE_DEFAULT};
pub use design::{firwin, FirWindow};
pub use poly::{resample_poly, resample_poly_len, ResampleFilter, RESAMPLE_POLY_DEFAULT_EDGE};
