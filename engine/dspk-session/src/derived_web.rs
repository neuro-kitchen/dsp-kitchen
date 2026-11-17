//! What a session derives from a recording, in a browser: there is no folder next to the
//! recording to keep it in, so nothing is read and saving says so. The names are those of the
//! desktop module (`derived/`).

use std::path::{Path, PathBuf};

use dspk_core::{DspError, DspResult};

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

fn no_files(what: &str) -> DspError {
    DspError::UnsupportedFormat(format!("{what} cannot be saved from a browser yet"))
}

pub fn ensure(_recording: &Path, _groups: &[&str]) -> DspResult<PathBuf> {
    Err(no_files("derived data"))
}

/// The session metadata, in a browser: nothing kept, the file's own not read (saving says so).
pub mod session_meta {
    use super::*;
    use crate::session_meta::SessionMetadata;

    pub const ATTRIBUTE: &str = "nwb_session";

    pub fn read(_recording: &Path) -> SessionMetadata {
        SessionMetadata::default()
    }

    pub fn write(_recording: &Path, _meta: &SessionMetadata) -> DspResult<PathBuf> {
        Err(no_files("the session's metadata"))
    }

    pub fn from_nwb_zarr(_dir: &Path) -> SessionMetadata {
        SessionMetadata::default()
    }
}

pub mod epochs {
    use super::*;
    use crate::epoch::EpochDoc;

    pub fn save(_recording: &Path, _epochs: &[EpochDoc]) -> DspResult<PathBuf> {
        Err(no_files("epochs"))
    }

    /// No epochs are stored for a recording in a browser.
    pub fn load(_recording: &Path) -> DspResult<Vec<EpochDoc>> {
        Ok(Vec::new())
    }
}

pub mod events {
    use super::*;

    /// One event, in seconds from the recording's start.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Event {
        pub channel: u32,
        pub onset_s: f64,
        /// `None`: the run ended inside the event.
        pub offset_s: Option<f64>,
        pub peak_value: f32,
    }

    pub fn read(_dir: &Path) -> DspResult<Vec<Event>> {
        Ok(Vec::new())
    }

    pub fn write(_recording: &Path, _name: &str, _events: &[Event], _names: &[String], _sample_rate: f64) -> DspResult<PathBuf> {
        Err(no_files("events"))
    }
}
