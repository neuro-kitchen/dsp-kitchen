//! Every compiled-in [`Format`], in detection order.
//!
//! The one place that names every generic format and holds their feature gates. Adding a format is
//! a new folder implementing [`Format`] plus one line here. Crates with more formats (e.g.
//! `dspk-synapse-io`) keep their own list and detect with [`crate::core::detect_in`].

use crate::core::Format;

static FORMATS: &[&dyn Format] = &[
    #[cfg(feature = "zarr")]
    &crate::generic::zarr_traces::ZarrTraces,
    &crate::generic::raw::Raw,
];

/// Compiled-in formats; the first whose [`Format::detect`] accepts a path reads it.
pub fn formats() -> &'static [&'static dyn Format] {
    FORMATS
}
