//! What the app derives from a recording lives in one Zarr store next to it,
//! `<recording>.dspk.zarr`, organized in groups as dspk's other Zarr outputs (`.nwb.zarr`,
//! `.sorting.zarr`), instead of many files beside the data:
//!
//! ```text
//! rec.bin
//! rec.dspk.zarr/               root attributes: dspk_derived (version), recording
//!   epochs/                    start_s, end_s; ids, names, kinds, notes as attributes (data::epochs)
//!   events/<name>/             an event set: channel, onset_sample, offset_sample, peak_sample,
//!                              peak_value; sample rate and channel names as attributes (data::events)
//! ```
//!
//! Sortings stay in their own stores (`.sorting.zarr`, Phy folders, NWB), as those formats lay out.
//! Python reads every group with `zarr.open`.

use std::path::{Path, PathBuf};

use dspk_core::DspResult;
use dspk_io::container::zarr::{open_rw_store, read_node_json, write_group};
use serde_json::{json, Map};

/// Format version of the store (`dspk_derived` attribute).
pub const DERIVED_VERSION: u32 = 1;

/// `rec.bin` → `rec.dspk.zarr` next to it (the recording's name without its data extension).
pub fn derived_store(recording: &Path) -> PathBuf {
    recording.with_file_name(format!("{}.dspk.zarr", stem(recording)))
}

/// The recording's file name without `.zarr`, `.nwb`, `.bin` or `.cbin` (and a trailing `/`).
pub fn stem(recording: &Path) -> String {
    let name = recording.file_name().map_or_else(|| "recording".into(), |n| n.to_string_lossy().into_owned());
    name.trim_end_matches('/').trim_end_matches(".zarr").trim_end_matches(".nwb").trim_end_matches(".bin").trim_end_matches(".cbin").to_string()
}

/// The folder of the recording's event sets (the store's `events` group).
pub fn events_dir(recording: &Path) -> PathBuf {
    derived_store(recording).join("events")
}

/// Event set `name` of the recording (`name` made safe for a group name).
pub fn events_group(recording: &Path, name: &str) -> PathBuf {
    let safe: String = name.chars().map(|c| if c.is_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' }).collect();
    events_dir(recording).join(if safe.is_empty() { "events".into() } else { safe })
}

/// Creates the store's root group (and `groups` under it, e.g. `"/events"`) when missing.
pub fn ensure(recording: &Path, groups: &[&str]) -> DspResult<PathBuf> {
    let dir = derived_store(recording);
    let store = open_rw_store(&dir)?;
    if read_node_json(&dir, "/").is_none() {
        let mut attrs = Map::new();
        attrs.insert("dspk_derived".into(), json!(DERIVED_VERSION));
        attrs.insert("recording".into(), json!(recording.file_name().map(|n| n.to_string_lossy().into_owned())));
        write_group(&store, &dir, "/", attrs)?;
    }
    for g in groups {
        if read_node_json(&dir, g).is_none() {
            write_group(&store, &dir, g, Map::new())?;
        }
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_of_a_recording() {
        let rec = Path::new("/data/m1/rec.ap.bin");
        assert_eq!(derived_store(rec), Path::new("/data/m1/rec.ap.dspk.zarr"));
        assert_eq!(derived_store(Path::new("/data/x.zarr/")), Path::new("/data/x.dspk.zarr"));
        assert_eq!(events_group(rec, "biceps onsets"), Path::new("/data/m1/rec.ap.dspk.zarr/events/biceps_onsets"));
    }
}
