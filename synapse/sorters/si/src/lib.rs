//! SpikeInterface's internal spike sorters, ported to GPU kernels (CubeCL): **SpyKING CIRCUS 2**
//! (feature `sc2`, [`sc2`]) and **Tridesclous 2** (feature `tdc2`, [`tdc2`]). Both are built from
//! SpikeInterface's `sortingcomponents`, ported in `dspk_synapse::sorter::components`; what they
//! share beyond those is in [`shared`]. MIT (SpikeInterface), distributed as MIT OR Apache-2.0.

pub mod shared;

#[cfg(feature = "sc2")]
pub mod sc2;
#[cfg(feature = "tdc2")]
pub mod tdc2;

#[cfg(feature = "sc2")]
pub use sc2::{Spykingcircus2Config, SPYKINGCIRCUS2_SORTER};
#[cfg(feature = "tdc2")]
pub use tdc2::{Tridesclous2Config, TRIDESCLOUS2_SORTER};
