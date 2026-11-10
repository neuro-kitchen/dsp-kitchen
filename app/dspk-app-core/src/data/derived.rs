//! What the app derives from a recording lives in one folder next to it, `<recording>.dspk/`,
//! instead of many files beside the data:
//!
//! ```text
//! rec.bin
//! rec.dspk/
//!   epochs.parquet          the recording's epochs (data::epochs)
//!   events/<name>.parquet   event sets: a crossings output, a selection of ticks (data::events)
//! ```
//!
//! Tables are Parquet (Polars / Arrow), so Python reads them with `polars.read_parquet`.

use std::path::{Path, PathBuf};

/// `rec.bin` → `rec.dspk` next to it (the recording's name without its data extension).
pub fn derived_dir(recording: &Path) -> PathBuf {
    recording.with_file_name(format!("{}.dspk", stem(recording)))
}

/// The recording's file name without `.zarr`, `.nwb`, `.bin` or `.cbin` (and a trailing `/`).
pub fn stem(recording: &Path) -> String {
    let name = recording.file_name().map_or_else(|| "recording".into(), |n| n.to_string_lossy().into_owned());
    name.trim_end_matches('/').trim_end_matches(".zarr").trim_end_matches(".nwb").trim_end_matches(".bin").trim_end_matches(".cbin").to_string()
}

/// The recording's epochs table.
pub fn epochs_file(recording: &Path) -> PathBuf {
    derived_dir(recording).join("epochs.parquet")
}

/// The folder of the recording's event sets.
pub fn events_dir(recording: &Path) -> PathBuf {
    derived_dir(recording).join("events")
}

/// Event set `name` of the recording (`name` made safe for a file name).
pub fn events_file(recording: &Path, name: &str) -> PathBuf {
    let safe: String = name.chars().map(|c| if c.is_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' }).collect();
    events_dir(recording).join(format!("{}.parquet", if safe.is_empty() { "events".into() } else { safe }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_of_a_recording() {
        let rec = Path::new("/data/m1/rec.ap.bin");
        assert_eq!(derived_dir(rec), Path::new("/data/m1/rec.ap.dspk"));
        assert_eq!(epochs_file(Path::new("/data/x.zarr/")), Path::new("/data/x.dspk/epochs.parquet"));
        assert_eq!(events_file(rec, "biceps onsets"), Path::new("/data/m1/rec.ap.dspk/events/biceps_onsets.parquet"));
    }
}
