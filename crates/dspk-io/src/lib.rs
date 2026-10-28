//! Recording file formats for `dspk`.
//!
//! Every format implements [`dspk_core::RecordingSource`], so callers read bounded chunks of any
//! recording without knowing how it is stored, and [`Format`], so [`open`] can pick the reader
//! from the path ([`registry::formats`]).
//!
//! - [`core`]: the [`Format`] trait, detection and opening, source listing, chunk cache.
//!   Out-of-core window streaming is `dspk_core::WindowLoader`.
//! - [`container`]: storage containers shared by every format (binary codecs, …).
//! - [`generic`]: domain-independent recording formats (raw binary, Zarr `/traces`).
//!
//! Neural-recording formats (NWB, SpikeGLX, mtscomp, Phy, probes) live in `dspk-synapse-io`.

pub mod container;
pub mod core;
pub mod generic;
pub mod registry;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

pub use crate::core::{
    default_source, detect, open, open_source, sources, CachedRecording, Format, MergedRecording, SourceEntry, SourceKind,
};
pub use generic::raw::{write_raw, RawParams, RawRecording};
#[cfg(feature = "zarr")]
pub use generic::zarr_traces::{write_zarr, ZarrRecording};
pub use registry::formats;
