//! Neural-recording formats (NWB, SpikeGLX, mtscomp), spike-sorting files (Phy / Kilosort, NWB
//! units, `.sorting.zarr`), probe geometry, and synthetic recordings, on top of `dspk-io`.
//!
//! [`open`], [`detect`], [`sources`] and [`open_source`] know every format of `dspk-io` plus the
//! neural ones ([`formats`]); `dspk_io::open` knows only the generic ones.

use std::path::Path;

use dspk_core::{DspResult, RecordingSource};
use dspk_io::{Format, SourceEntry};

pub mod mtscomp;
#[cfg(feature = "zarr")]
pub mod nwb;
pub mod phy;
pub mod probe;
mod sorting_format;
#[cfg(feature = "zarr")]
pub mod sorting_zarr;
pub mod spikeglx;
pub mod synthetic;
pub mod templates;

pub use mtscomp::MtscompRecording;
#[cfg(feature = "zarr")]
pub use nwb::NwbZarrRecording;
pub use probe::{probe_of, Position3D, SensorLayout, SensorSite};
pub use sorting_format::{detect_sorting, SortingFormat};
pub use spikeglx::SpikeGlxMeta;
pub use synthetic::{SyntheticParams, SyntheticRecording};

static FORMATS: &[&dyn Format] = &[
    // NWB before generic Zarr: an NWB store is also a Zarr store
    #[cfg(feature = "zarr")]
    &nwb::Nwb,
    #[cfg(feature = "zarr")]
    &dspk_io::generic::zarr_traces::ZarrTraces,
    // SpikeGLX before mtscomp: a SpikeGLX `.cbin` is read with its `.meta`
    &spikeglx::SpikeGlx,
    &mtscomp::Mtscomp,
    &dspk_io::generic::raw::Raw,
];

/// Every compiled-in format, neural and generic, in detection order.
pub fn formats() -> &'static [&'static dyn Format] {
    FORMATS
}

/// The format of `path` among [`formats`].
pub fn detect(path: &Path) -> DspResult<&'static dyn Format> {
    dspk_io::core::detect_in(formats(), path)
}

/// Opens the default source of `path` (see [`Format::open_default`]).
pub fn open(path: &Path) -> DspResult<Box<dyn RecordingSource>> {
    detect(path)?.open_default(path)
}

/// Every source of `path`, from metadata only.
pub fn sources(path: &Path) -> DspResult<Vec<SourceEntry>> {
    detect(path)?.sources(path)
}

/// Opens source `id` of `path` (an id from [`sources`]).
pub fn open_source(path: &Path, id: &str) -> DspResult<Box<dyn RecordingSource>> {
    detect(path)?.open(path, id)
}
