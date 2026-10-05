//! Unified Sorter Output Persistence & Interoperability (`storage/mod.rs`).
//!
//! Exposes auto-detecting [`save_sorting`] and [`load_sorting`] functions supporting:
//! - **Phy / Kilosort Flat Folder** (`.phy`, or directory with `spike_times.npy` / `params.py`)
//! - **Self-Contained Zarr Sorting Analyzer Store** (`.sorting.zarr`, `.zarr`)
//! - **NWB `/units` DynamicTable Group** inside `.nwb.zarr`
//!
//! [`load_spikes`] reads any of them spike by spike ([`PhySorting`]: what curation works on).

pub mod nwb_units;
pub mod phy;
pub mod phy_sorting;
pub mod zarr_analyzer;

use std::path::Path;
use dsp_core::{DspError, DspResult};
use crate::core::SortingOutput;

pub use nwb_units::{load_nwb_units, save_nwb_units};
pub use phy::{load_phy_folder, save_phy_folder};
pub use phy_sorting::{load_spikes, resolve_dat_path, ClusterTables, PhyParams, PhySorting, PhyTemplates};
pub use zarr_analyzer::{load_sorting_zarr, save_sorting_zarr};

/// Supported persistent disk formats for spike sorting outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortingFormat {
    /// Phy / Kilosort directory of `.npy` arrays and `.tsv` tables
    Phy,
    /// Self-contained `.sorting.zarr` analyzer store
    ZarrAnalyzer,
    /// NWB `/units` table inside `.nwb.zarr`
    NwbUnits,
}

/// Automatically detects format and saves [`SortingOutput`] to `path`.
pub fn save_sorting(sorting: &SortingOutput, path: &Path, format: Option<SortingFormat>) -> DspResult<()> {
    let fmt = format.unwrap_or_else(|| {
        let p_str = path.to_string_lossy();
        if p_str.ends_with(".nwb.zarr") {
            SortingFormat::NwbUnits
        } else if p_str.ends_with(".sorting.zarr") || p_str.ends_with(".zarr") {
            SortingFormat::ZarrAnalyzer
        } else {
            SortingFormat::Phy
        }
    });

    match fmt {
        SortingFormat::Phy => save_phy_folder(sorting, path),
        SortingFormat::ZarrAnalyzer => save_sorting_zarr(sorting, path),
        SortingFormat::NwbUnits => save_nwb_units(sorting, path),
    }
}

/// Automatically detects format and loads [`SortingOutput`] from `path`.
pub fn load_sorting(path: &Path) -> DspResult<SortingOutput> {
    if !path.exists() {
        return Err(DspError::Io(format!("Path {} does not exist", path.display())));
    }

    if path.join("spike_times.npy").exists() {
        return load_phy_folder(path);
    }
    if zarr_store::has_array(path, "/units/spike_times")
        || zarr_store::has_array(path, "/units/spike_times_index")
        || (zarr_store::has_array(path, "/spike_times") && zarr_store::has_array(path, "/spike_times_index"))
    {
        return load_nwb_units(path, None);
    }
    if zarr_store::has_array(path, "/spikes/times") && path.join("zarr.json").exists() {
        return load_sorting_zarr(path);
    }

    let p_str = path.to_string_lossy();
    if p_str.ends_with(".nwb.zarr") {
        return load_nwb_units(path, None);
    }
    if p_str.ends_with(".sorting.zarr") || p_str.ends_with(".zarr") {
        return load_sorting_zarr(path);
    }

    // Default fallback to Phy folder
    load_phy_folder(path)
}
