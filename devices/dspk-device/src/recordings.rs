//! Recording folders: one per `start` that records, named by its start time (UTC) and label, and
//! recovery at start-up of the ones a crash left open.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use dspk_core::RecordingSource;
use dspk_io::live::{SessionLog, RAW_ZARR};
use serde_json::json;

/// `2026-11-17T15-04-05Z` for `unix_s` (file names cannot hold `:` everywhere).
pub fn folder_time(unix_s: u64) -> String {
    let days = (unix_s / 86_400) as i64;
    let secs = unix_s % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}-{:02}-{:02}Z", secs / 3600, secs / 60 % 60, secs % 60)
}

/// Days since 1970-01-01 → (year, month, day) (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// A label made safe for a folder name: letters, digits, `-` and `_` kept, others `_`, at most 40.
pub fn safe_label(label: &str) -> String {
    label.chars().filter(|c| !c.is_control()).map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).take(40).collect()
}

/// A new recording's folder under `dir` (never an existing one: a numeric suffix when the second
/// is taken).
pub fn new_folder(dir: &Path, label: &str) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let label = safe_label(label);
    let base = if label.is_empty() { folder_time(now) } else { format!("{}_{label}", folder_time(now)) };
    for n in 0..1000 {
        let name = if n == 0 { base.clone() } else { format!("{base}-{n}") };
        let path = dir.join(&name);
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::other(format!("{}: no free folder name for {base}", dir.display())))
}

/// Recordings under `dir` whose session log has no end (a crash): each gets a `recovered` line
/// with the samples its Zarr holds. Returns the folders recovered.
pub fn recover(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut recovered = Vec::new();
    for entry in entries.flatten() {
        let folder = entry.path();
        if !folder.join(dspk_io::live::SESSION_LOG_FILE).exists() || SessionLog::ended_cleanly(&folder) {
            continue;
        }
        let samples = dspk_io::ZarrRecording::open(&folder.join(RAW_ZARR)).map(|r| r.info().samples).ok();
        let last_heartbeat = SessionLog::read(&folder).ok().and_then(|(entries, _)| entries.into_iter().rev().find(|e| e.kind == "heartbeat"));
        match SessionLog::append_to(&folder) {
            Ok(mut log) => {
                let note = "the device stopped without ending this recording (a crash or a power loss); the data ends at the last chunk stored";
                let _ = log.log("recovered", json!({"samples_stored": samples, "last_heartbeat": last_heartbeat, "note": note}));
                tracing::warn!("recovered {}: {} samples stored", folder.display(), samples.unwrap_or(0));
                recovered.push(folder);
            }
            Err(e) => tracing::warn!("{}: cannot mark it recovered: {e}", folder.display()),
        }
    }
    recovered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folder_names() {
        assert_eq!(folder_time(0), "1970-01-01T00-00-00Z");
        // 2026-11-17 15:04:05 UTC
        assert_eq!(folder_time(1_794_927_845), "2026-11-17T15-04-05Z");
        assert_eq!(safe_label("PPG / left hand"), "PPG___left_hand");
    }
}
