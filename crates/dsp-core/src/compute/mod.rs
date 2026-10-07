//! The compute layer (`compute` feature): the single entry point through which every crate reaches
//! a CubeCL runtime.
//!
//! - [`ComputeTask`] / [`ComputeTarget::run`](crate::device::ComputeTarget): run generic work on
//!   the selected runtime.
//! - [`LaunchGeometry`]: cube sizes and counts from the runtime's properties.
//! - [`tune`]: CubeCL autotuning for launch settings with no device-independent best value.
//! - [`bench`]: device-synchronised timing.
//!
//! Algorithm crates (`dsp-base`, `dsp-synapse`, …) take a `Client` and use these; they
//! never inspect or special-case the device.

pub mod bench;
pub mod launch;
pub mod limits;
mod special;
mod target;
pub mod tune;

pub use crate::device::{ComputeError, ComputeTarget, RUNTIME_ENV};
pub use limits::{device_elements, MAX_DEVICE_ELEMENTS};
pub use launch::{LaunchGeometry, channel_position, row_position, sample_position};
pub use special::{negative_infinity, positive_infinity, NEGATIVE_INFINITY_BITS, POSITIVE_INFINITY_BITS};
pub use target::ComputeTask;
