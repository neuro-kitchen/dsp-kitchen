//! Every compiled-in [`Format`], in detection order.
//!
//! The one place that names every generic format and holds their feature gates. Adding a format is
//! a new folder implementing [`Format`] plus one line here. Crates with more formats (e.g.
//! `dspk_synapse::io`) keep their own list and detect with [`crate::core::detect_in`].

use crate::core::Format;

static FORMATS: &[&dyn Format] = &[
    // Folders on disk: not in a browser page
    #[cfg(all(feature = "zarr", not(target_family = "wasm")))]
    &crate::generic::zarr_traces::ZarrTraces,
    &crate::generic::wav::Wav,
    &crate::generic::raw::Raw,
];

/// Compiled-in formats; the first whose [`Format::detect`] accepts a path reads it.
pub fn formats() -> &'static [&'static dyn Format] {
    FORMATS
}
