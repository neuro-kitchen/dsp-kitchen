//! Device-side building blocks shared by every dsp-base algorithm.
//!
//! Runtime selection and launch geometry live in [`dsp_core::compute`]; this module holds what the
//! kernels of this crate have in common:
//!
//! - [`DspFloat`]: the element type kernels are generic over.
//! - [`buffer`]: allocate, upload and download typed device buffers (sizes from the element type).
//! - [`Scratch`]: a device buffer that grows on demand and is reused across calls.
//! - [`EdgeMode`] and [`read_extended`]: how stencils read samples past either end of a row.
//! - [`layout`]: transposes between channel-major and time-major buffers.
//! - [`reduce`]: per-row reductions (mean / standard deviation, k-th smallest |x| by radix select),
//!   one cube per row; their kernels live in [`kernels`].

pub mod buffer;
pub mod edge;
pub mod kernels;
mod float;
pub mod layout;
pub mod reduce;
mod scratch;

pub use edge::EdgeMode;
pub use kernels::{read_extended, read_extended_strided};
pub use float::{cast, cast_all, cast_f32, to_f64, DspFloat};
pub use scratch::Scratch;
