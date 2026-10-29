//! `dspk-sort`: what every dspk spike sorter shares, so each sorter crate (`dspk-ks4`, `dspk-ms5`,
//! `dspk-sc2`, `dspk-tdc2`) holds only its own algorithm, under its upstream's licence.
//!
//! - [`result`]: the common result of a sort ([`SorterResult`]) and its conversion to a
//!   `SortingOutput`.
//! - [`windows`]: recording windows streamed to the device and preprocessed there
//!   ([`DeviceWindows`]), with an optional per-window step ([`WindowAligner`]).
//! - [`stats`]: NumPy's median and other small helpers.
//! - [`components`]: sorter building blocks ported from SpikeInterface's `sortingcomponents`
//!   (MIT): matched-filter detection, circus-omp and tdc peelers, splits, merges, templates.
//!
//! This crate is MIT OR Apache-2.0; it never depends on a GPL crate.

pub mod components;
pub mod result;
pub mod stats;
pub mod windows;

pub use result::{AmplitudeScale, SorterResult};
pub use windows::{snippet_fits, spread_windows, DeviceWindows, WindowAligner};
