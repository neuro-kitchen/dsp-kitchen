//! Sample-rate conversion on the device, following `scipy.signal`:
//!
//! - [`resample_poly`]: rational `up / down` resampling with a polyphase FIR (`resample_poly`).
//! - [`decimate`]: integer down-sampling behind an anti-aliasing filter (`decimate`).
//! - [`design`]: the windowed-sinc FIR design both use (`firwin`).
//! - [`fractional`]: fractional-delay (sub-sample shift) interpolation with a windowed sinc, host
//!   and device.

pub mod decimate;
pub mod design;
pub mod fractional;
pub mod kernels;
pub mod poly;

pub use decimate::{decimate, decimate_len, DecimateFilter, DECIMATE_DEFAULT};
pub use design::{firwin, FirWindow};
pub use fractional::{fractional_delay, fractional_delay_taps, fractional_delay_taps_device};
pub use poly::{resample_poly, resample_poly_len, ResampleFilter, RESAMPLE_POLY_DEFAULT_EDGE};
