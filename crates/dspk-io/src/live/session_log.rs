//! `session-live.jsonl`: everything needed to understand a recording afterwards, crash or not.
//!
//! JSON Lines: one complete object per line, appended and flushed per entry, synced to the disk at
//! least once a second and at every state change. A crash loses at most the last line; every line
//! before it still reads. A recording whose log has no `stop` line ended in a crash: the device
//! says so at its next start and appends a `recovered` line.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use dspk_core::{DspError, DspResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::unix_now_ns;

/// The log's file name in a recording's folder.
pub const SESSION_LOG_FILE: &str = "session-live.jsonl";

/// Longest time between syncs to the disk.
const SYNC_INTERVAL: Duration = Duration::from_secs(1);

/// One line. `type` names it; the other fields depend on it (the design note lists them: `start`,
/// `state`, `control`, `client`, `parameter`, `mark`, `gap`, `drop`, `clock`, `heartbeat`, `log`,
/// `stop`, `recovered`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEntry {
    #[serde(rename = "type")]
    pub kind: String,
    /// When it was written (Unix ns).
    pub unix_ns: u64,
    #[serde(flatten)]
    pub fields: serde_json::Map<String, Value>,
}

impl SessionEntry {
    /// An entry of `kind` with the fields of `fields` (a JSON object; anything else is kept under
    /// `value`).
    pub fn new(kind: &str, fields: Value) -> Self {
        let fields = match fields {
            Value::Object(map) => map,
            Value::Null => serde_json::Map::new(),
            other => serde_json::Map::from_iter([("value".to_string(), other)]),
        };
        Self { kind: kind.to_string(), unix_ns: unix_now_ns(), fields }
    }

    /// Entries that are synced to the disk at once.
    fn is_milestone(&self) -> bool {
        matches!(self.kind.as_str(), "start" | "state" | "stop" | "recovered" | "control")
    }
}

pub struct SessionLog {
    path: PathBuf,
    file: File,
    last_sync: Instant,
}

impl SessionLog {
    /// Creates the log in `folder` (a new recording's folder); fails if one exists there already.
    pub fn create(folder: &Path) -> DspResult<Self> {
        let path = folder.join(SESSION_LOG_FILE);
        let file = OpenOptions::new().create_new(true).append(true).open(&path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
        Ok(Self { path, file, last_sync: Instant::now() })
    }

    /// Opens an existing log to add to it (a recovered recording). A last line cut short by a crash
    /// is ended first, so the next entry starts a line of its own.
    pub fn append_to(folder: &Path) -> DspResult<Self> {
        let path = folder.join(SESSION_LOG_FILE);
        let io = |e: std::io::Error| DspError::Io(format!("{}: {e}", path.display()));
        let ends_cut = std::fs::read(&path).map_err(io)?.last().is_some_and(|&b| b != b'\n');
        let mut file = OpenOptions::new().append(true).open(&path).map_err(io)?;
        if ends_cut {
            file.write_all(b"\n").map_err(io)?;
        }
        Ok(Self { path, file, last_sync: Instant::now() })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends one entry.
    pub fn write(&mut self, entry: &SessionEntry) -> DspResult<()> {
        let mut line = serde_json::to_vec(entry).map_err(|e| DspError::Io(format!("{}: {e}", self.path.display())))?;
        line.push(b'\n');
        let io = |e: std::io::Error| DspError::Io(format!("{}: {e}", self.path.display()));
        self.file.write_all(&line).map_err(io)?;
        if entry.is_milestone() || self.last_sync.elapsed() >= SYNC_INTERVAL {
            self.file.sync_data().map_err(io)?;
            self.last_sync = Instant::now();
        }
        Ok(())
    }

    /// Appends an entry of `kind` with `fields`.
    pub fn log(&mut self, kind: &str, fields: Value) -> DspResult<()> {
        self.write(&SessionEntry::new(kind, fields))
    }

    /// Every entry that reads in `folder`'s log, and whether the last line was cut short (a crash
    /// in the middle of writing it).
    pub fn read(folder: &Path) -> DspResult<(Vec<SessionEntry>, bool)> {
        let path = folder.join(SESSION_LOG_FILE);
        let file = File::open(&path).map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
        let mut entries = Vec::new();
        let mut cut = false;
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|e| DspError::Io(format!("{}: {e}", path.display())))?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str(&line) {
                Ok(entry) => {
                    entries.push(entry);
                    cut = false;
                }
                Err(_) => cut = true,
            }
        }
        Ok((entries, cut))
    }

    /// Whether the recording in `folder` ended cleanly (its log has a `stop` or `recovered` line).
    pub fn ended_cleanly(folder: &Path) -> bool {
        Self::read(folder).is_ok_and(|(entries, _)| entries.iter().any(|e| e.kind == "stop" || e.kind == "recovered"))
    }
}

impl Drop for SessionLog {
    fn drop(&mut self) {
        let _ = self.file.sync_data();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn appends_reads_back_and_survives_a_cut_line() {
        let dir = std::env::temp_dir().join(format!("dspk_session_log_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        {
            let mut log = SessionLog::create(&dir).unwrap();
            log.log("start", json!({"device": "test"})).unwrap();
            log.log("heartbeat", json!({"head_sample": 30_000})).unwrap();
            assert!(SessionLog::create(&dir).is_err(), "one log per folder");
        }
        assert!(!SessionLog::ended_cleanly(&dir));
        // A crash in the middle of a line
        let mut f = OpenOptions::new().append(true).open(dir.join(SESSION_LOG_FILE)).unwrap();
        f.write_all(b"{\"type\":\"heartb").unwrap();
        let (entries, cut) = SessionLog::read(&dir).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(cut);
        assert_eq!(entries[1].fields["head_sample"], json!(30_000));

        SessionLog::append_to(&dir).unwrap().log("recovered", json!({"head_sample": 30_000})).unwrap();
        assert!(SessionLog::ended_cleanly(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
