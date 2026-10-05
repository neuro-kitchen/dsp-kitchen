//! Which sorting file format a path holds.

use std::path::Path;

use crate::neuro::phy::PhyFolder;
#[cfg(feature = "zarr")]
use crate::neuro::{nwb::NwbUnitsTable, sorting_zarr::SortingZarr};

/// Spike-sorting file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortingFormat {
    /// Phy / Kilosort folder of `.npy` arrays and `.tsv` tables.
    Phy,
    /// dsp-kitchen `.sorting.zarr` store.
    SortingZarr,
    /// NWB `/units` table in a `.nwb.zarr` store.
    NwbUnits,
}

impl SortingFormat {
    /// The format a path's name implies: `.nwb.zarr` → NWB units, other `.zarr` → `.sorting.zarr`,
    /// anything else → Phy (the format to write when none is asked for).
    pub fn from_path_name(path: &Path) -> Self {
        let name = path.to_string_lossy();
        if name.ends_with(".nwb.zarr") {
            Self::NwbUnits
        } else if name.ends_with(".zarr") {
            Self::SortingZarr
        } else {
            Self::Phy
        }
    }
}

/// The format of the sorting at `path`, from its contents, else from its name
/// ([`SortingFormat::from_path_name`]).
pub fn detect_sorting(path: &Path) -> SortingFormat {
    if PhyFolder::is_phy_folder(path) {
        return SortingFormat::Phy;
    }
    #[cfg(feature = "zarr")]
    {
        if NwbUnitsTable::is_units_table(path) {
            return SortingFormat::NwbUnits;
        }
        if SortingZarr::is_sorting_zarr(path) {
            return SortingFormat::SortingZarr;
        }
    }
    SortingFormat::from_path_name(path)
}
