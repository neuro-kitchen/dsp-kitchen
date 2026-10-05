//! Every compiled-in [`Format`], in detection order.
//!
//! The one place that names every format and holds their feature gates. Adding a format is a
//! new folder implementing [`Format`] plus one line here.

use crate::core::Format;

static FORMATS: &[&dyn Format] = &[
    // NWB before generic Zarr: an NWB store is also a Zarr store
    #[cfg(all(feature = "neuro", feature = "zarr"))]
    &crate::neuro::nwb::Nwb,
    #[cfg(feature = "zarr")]
    &crate::generic::zarr_traces::ZarrTraces,
    // SpikeGLX before mtscomp: a SpikeGLX `.cbin` is read with its `.meta`
    #[cfg(feature = "neuro")]
    &crate::neuro::spikeglx::SpikeGlx,
    #[cfg(feature = "neuro")]
    &crate::neuro::mtscomp::Mtscomp,
    &crate::generic::raw::Raw,
];

/// Compiled-in formats; the first whose [`Format::detect`] accepts a path reads it.
pub fn formats() -> &'static [&'static dyn Format] {
    FORMATS
}
