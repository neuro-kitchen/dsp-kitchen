//! Workspace persistence: dock tree, view settings, panel visibility and recent files.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::dock::Dock;
use super::view_state::ViewState;

/// Which side / bottom panels are shown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Panels {
    pub data: bool,
    pub properties: bool,
    pub timeline: bool,
}

impl Default for Panels {
    fn default() -> Self {
        Self { data: true, properties: true, timeline: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub version: u32,
    pub dock: Dock,
    pub views: Vec<ViewState>,
    pub focused: Option<u32>,
    pub panels: Panels,
    pub recent: Vec<PathBuf>,
    pub window_sec: f64,
}

impl Session {
    pub const VERSION: u32 = 1;

    /// The layout is usable when every view in the tree exists, ids are unique, and every
    /// selected channel exists in a dataset with `channels` channels.
    pub fn is_valid_for(&self, channels: usize) -> bool {
        let tree = self.dock.views();
        let mut ids: Vec<u32> = self.views.iter().map(|v| v.id).collect();
        ids.sort_unstable();
        ids.dedup();
        self.version == Self::VERSION
            && ids.len() == self.views.len()
            && tree.iter().all(|id| ids.binary_search(id).is_ok())
            && self.views.iter().all(|v| v.selection.iter().all(|&c| c < channels))
    }

    /// `$XDG_CONFIG_HOME/croc-app/session.json` (or `~/.config/…`; `%APPDATA%` on Windows).
    pub fn path() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from))
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
        Some(base.join("croc-app").join("session.json"))
    }

    pub fn load() -> Option<Session> {
        let text = std::fs::read_to_string(Self::path()?).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = Self::path() else { return Ok(()) };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }
}
