//! Sortings on disk ↔ [`SortingOutput`]. The files are read and written by `dsp-io`
//! (`dsp_io::neuro::{phy, nwb, sorting_zarr}`); this module converts them:
//! - **Phy / Kilosort** folder (directory with `spike_times.npy`)
//! - **`.sorting.zarr`**: dsp-kitchen's own store
//! - **NWB `/units`** table in a `.nwb.zarr` store
//!
//! [`load_spikes`] reads any of them spike by spike (a `PhyFolder`: what curation works on).

pub mod nwb_units;
pub mod phy;
pub mod sorting_zarr;

use std::path::Path;

use dsp_core::{DspError, DspResult};
use dsp_io::neuro::{detect_sorting, SortingFormat};

use crate::core::SortingOutput;

pub use nwb_units::{load_nwb_units, save_nwb_units};
pub use phy::{fill_similarity, load_phy_folder, load_spikes, save_phy_folder};
pub use sorting_zarr::{load_sorting_zarr, save_sorting_zarr};

/// Saves `sorting` at `path` as `format`, or as the format the path's name implies
/// ([`SortingFormat::from_path_name`]).
pub fn save_sorting(sorting: &SortingOutput, path: &Path, format: Option<SortingFormat>) -> DspResult<()> {
    match format.unwrap_or_else(|| SortingFormat::from_path_name(path)) {
        SortingFormat::Phy => save_phy_folder(sorting, path),
        SortingFormat::SortingZarr => save_sorting_zarr(sorting, path),
        SortingFormat::NwbUnits => save_nwb_units(sorting, path),
    }
}

/// Loads the sorting at `path`, in the format detected from its contents (else its name).
pub fn load_sorting(path: &Path) -> DspResult<SortingOutput> {
    if !path.exists() {
        return Err(DspError::Io(format!("Path {} does not exist", path.display())));
    }
    match detect_sorting(path) {
        SortingFormat::Phy => load_phy_folder(path),
        SortingFormat::SortingZarr => load_sorting_zarr(path),
        SortingFormat::NwbUnits => load_nwb_units(path, None),
    }
}
