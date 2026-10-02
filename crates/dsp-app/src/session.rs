//! What the app remembers between runs: recent recordings (newest first), the workspace shown.
//!
//! Its own file, `$XDG_CONFIG_HOME/dsp-app/workbench.json`, so the Slint app (which keeps
//! `session.json` in the same folder) and this one never overwrite each other. On a first run the
//! recent list is taken from the Slint app's file.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::workspace::Workspace;

/// Recent recordings kept.
const RECENT: usize = 10;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub version: u32,
    #[serde(default)]
    pub recent: Vec<PathBuf>,
    #[serde(default)]
    pub workspace: Workspace,
}

impl Session {
    pub const VERSION: u32 = 1;

    pub fn new() -> Self {
        Self { version: Self::VERSION, ..Self::default() }
    }

    fn config_dir() -> Option<PathBuf> {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from))
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    }

    pub fn path() -> Option<PathBuf> {
        Some(Self::config_dir()?.join("dsp-app").join("workbench.json"))
    }

    /// The saved session; else the Slint app's recent files; else an empty one.
    pub fn load(path: Option<&Path>) -> Self {
        let read = |p: &Path| std::fs::read_to_string(p).ok();
        if let Some(s) = path.and_then(read).and_then(|t| serde_json::from_str::<Session>(&t).ok()).filter(|s| s.version == Self::VERSION) {
            return s;
        }
        let slint = path.and_then(|p| p.parent()).map(|d| d.join("session.json"));
        let recent = slint
            .as_deref()
            .and_then(read)
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| serde_json::from_value::<Vec<PathBuf>>(v.get("recent")?.clone()).ok())
            .unwrap_or_default();
        Self { recent, ..Self::new() }
    }

    pub fn save(&self, path: Option<&Path>) -> std::io::Result<()> {
        let Some(path) = path else { return Ok(()) };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self).map_err(std::io::Error::other)?)
    }

    /// `path` becomes the newest recent recording.
    pub fn push_recent(&mut self, path: &Path) {
        self.recent.retain(|p| p != path);
        self.recent.insert(0, path.to_path_buf());
        self.recent.truncate(RECENT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("dsp-app-session-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn test_round_trip_and_recent_order() {
        let dir = scratch("round");
        let path = dir.join("workbench.json");
        let mut s = Session::new();
        for p in ["/a", "/b", "/a"] {
            s.push_recent(Path::new(p));
        }
        assert_eq!(s.recent, vec![PathBuf::from("/a"), PathBuf::from("/b")]);
        s.workspace = Workspace::Curation;
        s.save(Some(&path)).unwrap();
        assert_eq!(Session::load(Some(&path)), s);
    }

    #[test]
    fn test_first_run_takes_the_slint_apps_recent_files() {
        let dir = scratch("slint");
        std::fs::write(dir.join("session.json"), r#"{"version": 3, "recent": ["/data/x.bin"], "active": "Time"}"#).unwrap();
        let s = Session::load(Some(&dir.join("workbench.json")));
        assert_eq!(s.recent, vec![PathBuf::from("/data/x.bin")]);
        assert_eq!(s.workspace, Workspace::Explore);
    }
}
